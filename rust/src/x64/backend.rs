// This module provides the x86-64 backend implementation that bridges the generic compiler
// infrastructure with x86-64 specific code generation. X64Backend implements the Backend trait
// to handle instruction selection and register allocation for x86-64 targets. It integrates
// ValueAssignmentManager for SSA value tracking, RegisterFile for physical register allocation,
// CompilerContext for coordinating value references, and InstructionSelector for emitting
// machine code. The backend demonstrates compilation of binary operations (add, sub) using
// register allocation with operand reuse optimization, generates function prologue/epilogue
// with proper stack frame setup, and provides a simplified instruction dispatch based on
// operand/result counts. This is a demonstration implementation showing how TPDE components
// work together; a production version would use proper opcode-based dispatch from the IR adaptor.

//! x86-64 Backend implementation.
//!
//! This module provides a complete x86-64 backend that bridges the CompilerBase
//! infrastructure with the x64_encoder. It demonstrates end-to-end compilation
//! from SSA IR through register allocation to machine code generation.

use super::encoder::{EncodingError, InstructionSelector};
use crate::core::{
    assembler::ElfAssembler,
    register_file::{RegAllocError, RegBitSet, RegisterFile},
    session::CompilationSession,
    value_assignment::ValueAssignmentManager,
    Backend, CompilerBase, CompilerContext, IrAdaptor, ValuePartRef, ValueRefError,
};
use bumpalo::Bump;

/// Error types for the x86-64 backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X64BackendError {
    /// Register allocation failed.
    RegisterAllocation(RegAllocError),
    /// Value reference operation failed.
    ValueRef(ValueRefError),
    /// Instruction encoding failed.
    Encoding(EncodingError),
    /// Unsupported instruction type.
    UnsupportedInstruction,
    /// Invalid operand configuration.
    InvalidOperands,
}

impl From<RegAllocError> for X64BackendError {
    fn from(err: RegAllocError) -> Self {
        Self::RegisterAllocation(err)
    }
}

impl From<ValueRefError> for X64BackendError {
    fn from(err: ValueRefError) -> Self {
        Self::ValueRef(err)
    }
}

impl From<EncodingError> for X64BackendError {
    fn from(err: EncodingError) -> Self {
        Self::Encoding(err)
    }
}

/// x86-64 backend implementing the Backend trait.
///
/// This backend demonstrates how to integrate all TPDE components:
/// - ValueAssignmentManager for tracking SSA values
/// - RegisterFile for register allocation
/// - ValueRef/ValuePartRef for instruction selection
/// - X64Encoder for machine code generation
pub struct X64Backend<'arena> {
    _alloc: &'arena Bump,
    /// Value assignment tracking.
    value_mgr: ValueAssignmentManager<'arena>,
    /// Register allocator.
    register_file: RegisterFile<'arena>,
    /// Instruction encoder.
    encoder: InstructionSelector<'arena>,
    /// Current stack frame size.
    frame_size: u32,
}

impl<'arena> X64Backend<'arena> {
    /// Create a new x86-64 backend.
    pub fn new(session: &'arena CompilationSession<'arena>) -> Result<Self, X64BackendError> {
        Ok(Self {
            _alloc: session.arena(),
            value_mgr: ValueAssignmentManager::new_in(session.arena()),
            register_file: RegisterFile::new(session, 16, 2, RegBitSet::all_in_bank(0, 16)),
            encoder: InstructionSelector::new(session.arena())?,
            frame_size: 0,
        })
    }

    /// Compile a binary arithmetic operation.
    fn compile_binary_op<A: IrAdaptor>(
        &mut self,
        base: &mut CompilerBase<A, ElfAssembler, Self>,
        inst: A::InstRef,
        op_type: BinaryOpType,
    ) -> Result<(), X64BackendError> {
        // Get operands - assume binary operation has exactly 2 operands + 1 result
        let operands: Vec<_> = base.adaptor.inst_operands(inst).collect();
        let results: Vec<_> = base.adaptor.inst_results(inst).collect();

        if operands.len() != 2 || results.len() != 1 {
            return Err(X64BackendError::InvalidOperands);
        }

        let left_val = operands[0];
        let right_val = operands[1];
        let result_val = results[0];

        // Create compiler context for managing borrows
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Get value references for operands
        let left_idx = base.adaptor.val_local_idx(left_val);
        let right_idx = base.adaptor.val_local_idx(right_val);
        let result_idx = base.adaptor.val_local_idx(result_val);

        let mut left_ref = ValuePartRef::new(left_idx, 0)?;
        let mut right_ref = ValuePartRef::new(right_idx, 0)?;
        let mut result_ref = ValuePartRef::new(result_idx, 0)?;

        // Load operands to registers
        let left_reg = left_ref.load_to_reg(&mut ctx)?;
        let right_reg = right_ref.load_to_reg(&mut ctx)?;

        // Try to reuse left operand register for result if possible
        let result_reg = result_ref.alloc_try_reuse(&mut left_ref, &mut ctx)?;

        // Emit the operation
        match op_type {
            BinaryOpType::Add => {
                self.encoder.compile_add(result_reg, left_reg, right_reg)?;
            }
            BinaryOpType::Sub => {
                self.encoder.compile_sub(result_reg, left_reg, right_reg)?;
            }
        }

        Ok(())
    }

    /// Compile a load from memory.
    fn compile_load<A: IrAdaptor>(
        &mut self,
        _base: &mut CompilerBase<A, ElfAssembler, Self>,
        _inst: A::InstRef,
    ) -> Result<(), X64BackendError> {
        // For now, just implement stack loads
        // In a real implementation, this would analyze the load instruction
        // and generate appropriate mov instructions
        Ok(())
    }

    /// Compile a store to memory.
    #[allow(dead_code)]
    fn compile_store<A: IrAdaptor>(
        &mut self,
        _base: &mut CompilerBase<A, ElfAssembler, Self>,
        _inst: A::InstRef,
    ) -> Result<(), X64BackendError> {
        // For now, just implement stack stores
        // In a real implementation, this would analyze the store instruction
        // and generate appropriate mov instructions
        Ok(())
    }

    /// Compile a return instruction.
    fn compile_return<A: IrAdaptor>(
        &mut self,
        _base: &mut CompilerBase<A, ElfAssembler, Self>,
        _inst: A::InstRef,
    ) -> Result<(), X64BackendError> {
        // Return is handled by epilogue
        Ok(())
    }
}

/// Binary operation types supported by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum BinaryOpType {
    Add,
    Sub,
}

impl<'arena, A: IrAdaptor> Backend<A, ElfAssembler> for X64Backend<'arena> {
    fn gen_prologue(&mut self, _base: &mut CompilerBase<A, ElfAssembler, Self>) {
        // Emit function prologue
        self.encoder.emit_prologue(self.frame_size).unwrap();
    }

    fn gen_epilogue(&mut self, _base: &mut CompilerBase<A, ElfAssembler, Self>) {
        // Emit function epilogue
        self.encoder.emit_epilogue(self.frame_size).unwrap();
    }

    fn compile_inst(
        &mut self,
        base: &mut CompilerBase<A, ElfAssembler, Self>,
        inst: A::InstRef,
    ) -> bool {
        // In a real implementation, this would dispatch based on instruction opcode
        // For demonstration, we'll assume specific instruction patterns

        // This is a simplified dispatch - real implementation would examine
        // instruction opcodes from the IR adaptor
        let operand_count = base.adaptor.inst_operands(inst).count();
        let result_count = base.adaptor.inst_results(inst).count();

        let result = match (operand_count, result_count) {
            (2, 1) => {
                // Binary operation - assume add for demonstration
                self.compile_binary_op(base, inst, BinaryOpType::Add)
            }
            (1, 0) => {
                // Unary operation - could be store or return
                self.compile_return(base, inst)
            }
            (0, 1) => {
                // Load operation
                self.compile_load(base, inst)
            }
            _ => Err(X64BackendError::UnsupportedInstruction),
        };

        match result {
            Ok(()) => true,
            Err(_err) => {
                // In a real implementation, we'd log the error
                false
            }
        }
    }
}

/// Helper function to create a complete x86-64 compilation pipeline.
pub fn create_x64_compiler<'arena, A: IrAdaptor>(
    adaptor: A,
    session: &'arena CompilationSession<'arena>,
) -> Result<CompilerBase<A, ElfAssembler, X64Backend<'arena>>, X64BackendError> {
    use crate::core::assembler::Assembler;
    let assembler = <ElfAssembler as Assembler<A>>::new(true);
    let backend = X64Backend::new(session)?;
    Ok(CompilerBase::new(adaptor, assembler, backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::IrAdaptor;
    use bumpalo::Bump;

    /// Minimal test IR adaptor for demonstrating the backend.
    struct TestIrAdaptor {
        functions: Vec<String>,
        current_func: Option<usize>,
        instructions: Vec<TestInstruction>,
        values: Vec<TestValue>,
    }

    #[derive(Debug, Clone)]
    struct TestInstruction {
        operands: Vec<usize>,
        results: Vec<usize>,
    }

    #[derive(Debug, Clone)]
    struct TestValue {
        local_idx: usize,
    }

    impl TestIrAdaptor {
        #[allow(dead_code)]
        fn new() -> Self {
            let mut adaptor = Self {
                functions: vec!["test_func".to_string()],
                current_func: None,
                instructions: Vec::new(),
                values: Vec::new(),
            };

            // Create a simple add operation: %2 = add %0, %1
            adaptor.values.push(TestValue { local_idx: 0 }); // %0
            adaptor.values.push(TestValue { local_idx: 1 }); // %1
            adaptor.values.push(TestValue { local_idx: 2 }); // %2

            adaptor.instructions.push(TestInstruction {
                operands: vec![0, 1], // %0, %1
                results: vec![2],     // %2
            });

            adaptor
        }
    }

    impl IrAdaptor for TestIrAdaptor {
        type ValueRef = usize;
        type InstRef = usize;
        type BlockRef = usize;
        type FuncRef = usize;

        const INVALID_VALUE_REF: Self::ValueRef = usize::MAX;
        const INVALID_BLOCK_REF: Self::BlockRef = usize::MAX;
        const INVALID_FUNC_REF: Self::FuncRef = usize::MAX;

        fn func_count(&self) -> u32 {
            self.functions.len() as u32
        }

        fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_> {
            Box::new(0..self.functions.len())
        }

        fn func_link_name(&self, func: Self::FuncRef) -> &str {
            &self.functions[func]
        }

        fn switch_func(&mut self, func: Self::FuncRef) -> bool {
            if func < self.functions.len() {
                self.current_func = Some(func);
                true
            } else {
                false
            }
        }

        fn reset(&mut self) {
            self.current_func = None;
        }

        fn entry_block(&self) -> Self::BlockRef {
            0
        }

        fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
            Box::new(std::iter::once(0))
        }

        fn block_insts(
            &self,
            _block: Self::BlockRef,
        ) -> Box<dyn Iterator<Item = Self::InstRef> + '_> {
            Box::new(0..self.instructions.len())
        }

        fn block_succs(
            &self,
            _block: Self::BlockRef,
        ) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
            Box::new(std::iter::empty())
        }

        fn inst_operands(
            &self,
            inst: Self::InstRef,
        ) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
            if inst < self.instructions.len() {
                Box::new(self.instructions[inst].operands.clone().into_iter())
            } else {
                Box::new(std::iter::empty())
            }
        }

        fn inst_results(
            &self,
            inst: Self::InstRef,
        ) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
            if inst < self.instructions.len() {
                Box::new(self.instructions[inst].results.clone().into_iter())
            } else {
                Box::new(std::iter::empty())
            }
        }

        fn val_local_idx(&self, val: Self::ValueRef) -> usize {
            if val < self.values.len() {
                self.values[val].local_idx
            } else {
                0
            }
        }

        fn val_ignore_liveness(&self, _val: Self::ValueRef) -> bool {
            false
        }

        fn set_block_idx(&self, _block: Self::BlockRef, _idx: usize) {
            // Not needed for test adaptor
        }
    }

    #[test]
    fn test_x64_backend_creation() {
        let arena = Box::leak(Box::new(Bump::new()));
        let session = Box::leak(Box::new(CompilationSession::new(arena)));
        let backend = X64Backend::new(session);
        assert!(backend.is_ok());
    }

    // NOTE: The end-to-end compilation tests have been removed as they relied on
    // the old generic trait-based architecture. The new concrete LlvmCompiler
    // implementation has its own comprehensive test suite in llvm/compiler.rs
    // and the integration tests in tests/llvm_compiler_tests.rs.
}
