//! TestIR compiler implementation for x86-64.
//!
//! This module provides a concrete compiler that can compile TestIR
//! to x86-64 machine code, matching the functionality of the C++ test harness.

use crate::{
    core::{
        assembler::{Assembler, ElfAssembler},
        register_file::{AsmReg, RegBitSet, RegisterFile},
        session::CompilationSession,
        value_assignment::ValueAssignmentManager,
        value_ref::ValuePartRef,
        CompilerContext, IrAdaptor,
    },
    test_ir::{adaptor::TestIRAdaptor, Operation, TestIR},
    x64::{
        encoder::EncodingError,
        function_codegen::{ArgInfo, FunctionCodegen},
    },
};
use log::debug;

/// TestIR compiler for x86-64.
pub struct TestIRCompiler<'arena> {
    /// The TestIR being compiled.
    ir: &'arena TestIR,
    /// TestIR adaptor.
    adaptor: TestIRAdaptor<'arena>,
    /// ELF assembler for object generation.
    assembler: ElfAssembler,
    /// Value assignment manager.
    value_mgr: ValueAssignmentManager<'arena>,
    /// Register allocator.
    register_file: RegisterFile,
    /// Compilation session.
    _session: &'arena CompilationSession<'arena>,
    /// Whether to disable fixed register assignments.
    _no_fixed_assignments: bool,
}

impl<'arena> TestIRCompiler<'arena> {
    /// Convert a global value index to a local index within the current function.
    fn global_to_local_idx(&self, global_idx: usize) -> usize {
        // Get the current function
        let func_idx = self.adaptor.cur_func() as usize;
        let func = &self.ir.functions[func_idx];

        // Local index is relative to the start of arguments for this function
        global_idx - func.arg_begin_idx as usize
    }

    /// Create a new TestIR compiler.
    pub fn new(
        ir: &'arena TestIR,
        session: &'arena CompilationSession<'arena>,
        no_fixed_assignments: bool,
    ) -> Result<Self, CompilationError> {
        let adaptor = TestIRAdaptor::new(ir);
        let assembler = <ElfAssembler as Assembler<TestIRAdaptor>>::new(true);

        // x86-64 has 16 general purpose registers
        let available_regs = RegBitSet::all_in_bank(0, 16);

        Ok(Self {
            ir,
            adaptor,
            assembler,
            value_mgr: ValueAssignmentManager::new_in(session.arena()),
            register_file: RegisterFile::new(16, 2, available_regs),
            _session: session,
            _no_fixed_assignments: no_fixed_assignments,
        })
    }

    /// Compile all functions in the IR.
    pub fn compile(mut self) -> Result<Vec<u8>, CompilationError> {
        for func_idx in 0..self.ir.functions.len() {
            self.compile_function(func_idx)?;
        }

        // Finalize and build object file
        <ElfAssembler as Assembler<TestIRAdaptor>>::finalize(&mut self.assembler);
        Ok(<ElfAssembler as Assembler<TestIRAdaptor>>::build_object_file(&mut self.assembler))
    }

    /// Compile a single function.
    fn compile_function(&mut self, func_idx: usize) -> Result<(), CompilationError> {
        let func = &self.ir.functions[func_idx];

        // Skip external declarations
        if func.declaration {
            return Ok(());
        }

        // Define symbol for function
        let sym = <ElfAssembler as Assembler<TestIRAdaptor>>::sym_predef_func(
            &mut self.assembler,
            &func.name,
            func.local_only,
            false,
        );

        // Switch adaptor to this function
        use crate::test_ir::adaptor::FuncRef;
        self.adaptor.switch_func(FuncRef(func_idx as u32));

        // Reset per-function state
        self.value_mgr = ValueAssignmentManager::new_in(self._session.arena());
        self.register_file = RegisterFile::new(16, 2, RegBitSet::all_in_bank(0, 16));

        // Create a new function codegen for this function
        let mut func_codegen = FunctionCodegen::new()?;

        // Process function arguments
        let arg_count = (func.arg_end_idx - func.arg_begin_idx) as usize;
        debug!(
            "Function {} has {} arguments (indices {}..{})",
            func.name, arg_count, func.arg_begin_idx, func.arg_end_idx
        );
        let args: Vec<ArgInfo> = vec![ArgInfo::int64(); arg_count];
        let arg_assignments = func_codegen.process_arguments(self._session, &args)?;

        // Assign arguments to registers/stack according to calling convention
        for (idx, assignment) in arg_assignments.iter().enumerate() {
            // Arguments start at local index 0 within the function
            let local_idx = idx;

            debug!("Creating argument assignment for local_idx={}", local_idx);

            // Create a value assignment for the argument
            let value_assignment = self.value_mgr.create_assignment(local_idx, 1, 8);

            // If argument is in a register, set it up
            if let Some(reg) = assignment.reg {
                // First, we need to actually allocate the register in the RegisterFile
                // We'll use allocate_specific_reg to mark it as allocated
                self.register_file
                    .allocate_specific_reg(reg, local_idx, 0)?;

                // Set up the part data for this register
                let part_data = &mut value_assignment.parts[0];
                *part_data = crate::core::value_assignment::PartData::new(reg.bank, reg.id, 3); // 3 = log2(8) for 64-bit
                part_data.set_register_valid(true);

                // Now we can lock the register since it's allocated
                self.register_file.lock_register(reg)?;
            } else {
                // TODO: Handle stack arguments - set up stack location
                value_assignment.location =
                    Some(crate::core::value_assignment::StorageLocation::Stack(
                        assignment.stack_off.unwrap_or(0),
                    ));
            }

            // Add initial reference for the argument
            value_assignment.add_ref();
        }

        // Set return value (always single i64 for TestIR)
        let rets = vec![ArgInfo::int64()];
        func_codegen.process_return_values(self._session, &rets)?;

        // Generate prologue
        func_codegen.emit_prologue()?;

        // Compile all blocks in the function
        for block_idx in func.block_begin_idx..func.block_end_idx {
            self.compile_block(block_idx as usize, &mut func_codegen)?;
        }

        // Generate epilogue (includes ret instruction)
        func_codegen.emit_epilogue()?;

        // Get generated code from function codegen
        let func_code = func_codegen.finalize()?;

        // Append to assembler
        let offset = self.assembler.append(&func_code, 16);
        let size = func_code.len() as u64;

        // Define symbol at this location
        self.assembler
            .define_symbol(sym, self.assembler.current_section(), offset, size);

        Ok(())
    }

    /// Compile a single block.
    fn compile_block(
        &mut self,
        block_idx: usize,
        func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        let block = &self.ir.blocks[block_idx];

        // Compile all instructions in the block
        for inst_idx in block.inst_begin_idx..block.inst_end_idx {
            self.compile_instruction(inst_idx as usize, func_codegen)?;
        }

        Ok(())
    }

    /// Compile a single instruction.
    fn compile_instruction(
        &mut self,
        inst_idx: usize,
        func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        let inst = &self.ir.values[inst_idx];

        let local_idx = self.global_to_local_idx(inst_idx);
        debug!("Compiling {} ({}): {:?}", inst.name, local_idx, inst.op);

        use Operation::*;
        match inst.op {
            Add => self.compile_add(inst_idx, func_codegen),
            Sub => self.compile_sub(inst_idx, func_codegen),
            Ret => self.compile_ret(inst_idx, func_codegen),
            Terminate => self.compile_terminate(func_codegen),
            Br | CondBr | Jump => self.compile_branch(inst_idx, func_codegen),
            Alloca => self.compile_alloca(inst_idx, func_codegen),
            Call => self.compile_call(inst_idx, func_codegen),
            Tbz => Err(CompilationError::UnsupportedInstruction), // ARM64-specific
            ZeroFill => self.compile_zerofill(inst_idx, func_codegen),
            Any => {
                // "Any" operations are just value definitions
                self.value_mgr.create_assignment(local_idx, 1, 8).add_ref();
                Ok(())
            }
            None if inst.value_type == crate::test_ir::ValueType::Phi => {
                // PHI nodes just need assignments
                self.value_mgr.create_assignment(local_idx, 1, 8).add_ref();
                Ok(())
            }
            None => Err(CompilationError::InvalidInstruction),
        }
    }

    /// Compile a binary arithmetic instruction.
    fn compile_binary_op<F>(
        &mut self,
        inst_idx: usize,
        func_codegen: &mut FunctionCodegen,
        emit_op: F,
    ) -> Result<(), CompilationError>
    where
        F: FnOnce(
            &mut crate::x64::encoder::X64Encoder,
            AsmReg,
            AsmReg,
        ) -> Result<(), EncodingError>,
    {
        let inst = &self.ir.values[inst_idx];

        // Get operands
        let left_idx = self.ir.value_operands[inst.op_begin_idx as usize];
        let right_idx = self.ir.value_operands[(inst.op_begin_idx + 1) as usize];

        // Convert to local indices
        let local_inst_idx = self.global_to_local_idx(inst_idx);
        let local_left = self.global_to_local_idx(left_idx as usize);
        let local_right = self.global_to_local_idx(right_idx as usize);

        // Create value assignment for the result
        self.value_mgr
            .create_assignment(local_inst_idx, 1, 8)
            .add_ref();

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Get value references
        let mut left_ref = ValuePartRef::new(local_left, 0)?;
        let mut right_ref = ValuePartRef::new(local_right, 0)?;
        let mut result_ref = ValuePartRef::new(local_inst_idx, 0)?;

        // Load operands
        let left_reg = left_ref.load_to_reg(&mut ctx)?;
        let right_reg = right_ref.load_to_reg(&mut ctx)?;

        // Allocate result (try to reuse left operand)
        let result_reg = result_ref.alloc_try_reuse(&mut left_ref, &mut ctx)?;

        // Generate instruction
        let encoder = func_codegen.encoder_mut();
        if result_reg != left_reg {
            encoder.mov_reg_reg(result_reg, left_reg)?;
        }
        emit_op(encoder, result_reg, right_reg)?;

        Ok(())
    }

    /// Compile an add instruction.
    fn compile_add(
        &mut self,
        inst_idx: usize,
        func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        self.compile_binary_op(inst_idx, func_codegen, |encoder, dst, src| {
            encoder.add_reg_reg(dst, src)
        })
    }

    /// Compile a sub instruction.
    fn compile_sub(
        &mut self,
        inst_idx: usize,
        func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        self.compile_binary_op(inst_idx, func_codegen, |encoder, dst, src| {
            encoder.sub_reg_reg(dst, src)
        })
    }

    /// Compile a return instruction.
    fn compile_ret(
        &mut self,
        inst_idx: usize,
        func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        let inst = &self.ir.values[inst_idx];

        // Get return value
        let ret_val_idx = self.ir.value_operands[inst.op_begin_idx as usize];

        // Convert to local index
        let local_ret = self.global_to_local_idx(ret_val_idx as usize);

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Get value reference
        let mut ret_ref = ValuePartRef::new(local_ret, 0)?;

        // Load to register
        let ret_reg = ret_ref.load_to_reg(&mut ctx)?;

        // Move to RAX if not already there
        let rax = AsmReg::new(0, 0);
        if ret_reg != rax {
            func_codegen.encoder_mut().mov_reg_reg(rax, ret_reg)?;
        }

        // Epilogue will handle the actual ret instruction
        Ok(())
    }

    /// Compile a terminate instruction.
    fn compile_terminate(
        &mut self,
        _func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        // For TestIR, terminate is like a void return
        // Epilogue will be generated at end of function
        Ok(())
    }

    /// Compile any branch instruction (br, condbr, jump).
    fn compile_branch(
        &mut self,
        _inst_idx: usize,
        _func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        // TODO: Implement control flow with label management and jump instructions
        Ok(())
    }

    /// Compile an alloca instruction.
    fn compile_alloca(
        &mut self,
        inst_idx: usize,
        _func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        // Create value assignment for the result (a pointer to stack space)
        let local_idx = self.global_to_local_idx(inst_idx);
        self.value_mgr.create_assignment(local_idx, 1, 8).add_ref();

        // For now, just allocate a dummy register
        // Real implementation would allocate stack space and return a pointer
        Ok(())
    }

    /// Compile a call instruction.
    fn compile_call(
        &mut self,
        inst_idx: usize,
        _func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        // Check if this call produces a result
        let inst = &self.ir.values[inst_idx];
        if inst.op.info().is_def {
            // Create value assignment for the result
            let local_idx = self.global_to_local_idx(inst_idx);
            self.value_mgr.create_assignment(local_idx, 1, 8).add_ref();
        }

        // For now, we don't handle the actual call generation
        Ok(())
    }

    /// Compile a zerofill instruction.
    fn compile_zerofill(
        &mut self,
        inst_idx: usize,
        func_codegen: &mut FunctionCodegen,
    ) -> Result<(), CompilationError> {
        let inst = &self.ir.values[inst_idx];

        // Check if zerofill has an operand
        if inst.op_begin_idx < inst.op_end_idx {
            let size = self.ir.value_operands[inst.op_begin_idx as usize];

            // Generate NOP instructions to fill space
            let encoder = func_codegen.encoder_mut();
            let nop_count = size / 4; // Approximate, each NOP is ~1-4 bytes

            for _ in 0..nop_count {
                encoder.nop()?;
            }
        }

        Ok(())
    }
}

/// Compilation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError {
    /// Encoding error from x64 encoder.
    Encoding(EncodingError),
    /// Invalid instruction in IR.
    InvalidInstruction,
    /// Unsupported instruction for this target.
    UnsupportedInstruction,
    /// Value reference error.
    ValueRef(crate::core::ValueRefError),
    /// Function codegen error.
    FunctionCodegen(crate::x64::FunctionCodegenError),
    /// Register allocation error.
    RegAlloc(crate::core::register_file::RegAllocError),
}

impl From<EncodingError> for CompilationError {
    fn from(err: EncodingError) -> Self {
        Self::Encoding(err)
    }
}

impl From<crate::core::ValueRefError> for CompilationError {
    fn from(err: crate::core::ValueRefError) -> Self {
        Self::ValueRef(err)
    }
}

impl From<crate::x64::FunctionCodegenError> for CompilationError {
    fn from(err: crate::x64::FunctionCodegenError) -> Self {
        Self::FunctionCodegen(err)
    }
}

impl From<crate::core::register_file::RegAllocError> for CompilationError {
    fn from(err: crate::core::register_file::RegAllocError) -> Self {
        Self::RegAlloc(err)
    }
}
