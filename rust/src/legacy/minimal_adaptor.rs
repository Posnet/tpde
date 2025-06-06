//! Basic LLVM IR adaptor implementation.
//!
//! This provides a minimal adaptor for walking LLVM modules using inkwell.

use inkwell::{
    basic_block::BasicBlock,
    module::Module,
    values::{BasicValueEnum, FunctionValue, InstructionValue},
};
use inkwell::llvm_sys::prelude::LLVMValueRef;
use inkwell::values::{AnyValue, AsValueRef};
use std::collections::HashMap;
use std::convert::TryFrom;
use crate::{core::{IrAdaptor, assembler::{Assembler, ElfAssembler}, CompilerBase, Backend}};

/// Adaptor walking an LLVM [`Module`] using `inkwell`.
///
/// This very small implementation only exposes functions which is enough for
/// [`CompilerBase::compile`].  Values, blocks and instructions are represented
/// as `Option` types so we can define an invalid constant.
pub struct LlvmIrAdaptor<'ctx> {
    funcs: Vec<FunctionValue<'ctx>>,
    names: Vec<String>,
    current: Option<FunctionValue<'ctx>>,
    val_index: HashMap<LLVMValueRef, usize>,
}

impl<'ctx> LlvmIrAdaptor<'ctx> {
    /// Create a new adaptor collecting all functions in the module.
    pub fn new(module: &Module<'ctx>) -> Self {
        let funcs: Vec<_> = module.get_functions().collect();
        let names = funcs
            .iter()
            .map(|f| f.get_name().to_str().unwrap_or("").to_string())
            .collect();
        Self {
            funcs,
            names,
            current: None,
            val_index: HashMap::new(),
        }
    }

    /// Switch to a different function and reset value indices.
    pub fn switch_func(&mut self, func: FunctionValue<'ctx>) -> bool {
        if !self.funcs.contains(&func) {
            return false;
        }
        self.current = Some(func);
        self.val_index.clear();

        // Build value index for all instructions in the function
        let mut idx = 0;
        for param in func.get_params() {
            self.val_index.insert(param.as_value_ref(), idx);
            idx += 1;
        }
        
        for bb in func.get_basic_blocks() {
            for inst in bb.get_instructions() {
                self.val_index.insert(inst.as_value_ref(), idx);
                idx += 1;
            }
        }
        
        true
    }

    /// Iterate over basic blocks in the current function.
    pub fn blocks(&self) -> impl Iterator<Item = Option<BasicBlock<'ctx>>> {
        self.current
            .map(|f| f.get_basic_blocks())
            .into_iter()
            .flatten()
            .map(Some)
    }

    /// Get a unique identifier for a basic block.
    pub fn block_id(&self, block: Option<BasicBlock<'ctx>>) -> usize {
        block
            .and_then(|b| {
                self.current?
                    .get_basic_blocks()
                    .iter()
                    .position(|&bb| bb == b)
            })
            .unwrap_or(0)
    }

    /// Get the value index if it exists.
    fn value_index(&self, val: LLVMValueRef) -> Option<usize> {
        self.val_index.get(&val).copied()
    }
}

/// Concrete definition of the invalid value constant.
const INVALID_VALUE: Option<BasicValueEnum> = None;

/// Concrete definition of the invalid block constant.
const INVALID_BLOCK: Option<BasicBlock> = None;


impl<'ctx> IrAdaptor for LlvmIrAdaptor<'ctx> {
    type FuncRef = Option<FunctionValue<'ctx>>;
    type ValueRef = Option<BasicValueEnum<'ctx>>;
    type BlockRef = Option<BasicBlock<'ctx>>;
    type InstRef = Option<InstructionValue<'ctx>>;

    const INVALID_VALUE_REF: Self::ValueRef = INVALID_VALUE;
    const INVALID_BLOCK_REF: Self::BlockRef = INVALID_BLOCK;
    const INVALID_FUNC_REF: Self::FuncRef = None;

    fn func_count(&self) -> u32 {
        self.funcs.len() as u32
    }

    fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_> {
        Box::new(self.funcs.iter().map(|&f| Some(f)))
    }

    fn func_link_name(&self, func: Self::FuncRef) -> &str {
        func.and_then(|f| {
            let idx = self.funcs.iter().position(|&ff| ff == f)?;
            self.names.get(idx).map(|s| s.as_str())
        })
        .unwrap_or("")
    }

    fn switch_func(&mut self, func: Self::FuncRef) -> bool {
        func.map(|f| self.switch_func(f)).unwrap_or(false)
    }

    fn reset(&mut self) {
        self.current = None;
        self.val_index.clear();
    }

    fn entry_block(&self) -> Self::BlockRef {
        self.current.and_then(|f| f.get_first_basic_block())
    }

    fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
        Box::new(self.blocks())
    }

    fn block_insts(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_> {
        if let Some(b) = block {
            Box::new(b.get_instructions().map(Some))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn block_succs(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
        if let Some(b) = block {
            if let Some(terminator) = b.get_terminator() {
                use inkwell::values::InstructionOpcode;
                match terminator.get_opcode() {
                    InstructionOpcode::Br => {
                        // Handle conditional and unconditional branches
                        let num_operands = terminator.get_num_operands();
                        if num_operands == 1 {
                            // Unconditional branch
                            Box::new(std::iter::once(terminator.get_operand(0).and_then(|op| op.right())))
                        } else if num_operands == 3 {
                            // Conditional branch
                            Box::new([1, 2].into_iter().filter_map(move |idx| {
                                terminator.get_operand(idx).and_then(|op| op.right())
                            }).map(Some))
                        } else {
                            Box::new(std::iter::empty())
                        }
                    },
                    InstructionOpcode::Switch => {
                        // Switch has default + cases
                        Box::new((1..terminator.get_num_operands()).step_by(2).filter_map(move |idx| {
                            terminator.get_operand(idx).and_then(|op| op.right())
                        }).map(Some))
                    },
                    _ => Box::new(std::iter::empty()),
                }
            } else {
                Box::new(std::iter::empty())
            }
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn inst_operands(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        if let Some(i) = inst {
            let num_operands = i.get_num_operands();
            Box::new((0..num_operands).filter_map(move |idx| {
                i.get_operand(idx)?.left().map(|v| Some(v))
            }))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn inst_results(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        if let Some(i) = inst {
            if i.get_type().is_void_type() {
                Box::new(std::iter::empty())
            } else {
                // Instructions that produce values have themselves as the result
                let val = BasicValueEnum::try_from(i.as_any_value_enum()).ok();
                Box::new(val.into_iter().map(Some))
            }
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn val_local_idx(&self, val: Self::ValueRef) -> usize {
        if let Some(v) = val {
            self.value_index(v.as_value_ref()).unwrap_or(0)
        } else {
            0
        }
    }

    fn val_ignore_liveness(&self, _val: Self::ValueRef) -> bool {
        false
    }
}

/// Simple x86-64 backend implementation.
#[allow(dead_code)]
pub struct X64Backend {
    reg_usage: u64,
}

impl X64Backend {
    pub fn new() -> Self {
        Self { reg_usage: 0 }
    }

    #[allow(dead_code)]
    fn emit_mov_imm32(&mut self, asm: &mut ElfAssembler, reg: u8, imm: u32) {
        // MOV r32, imm32: B8+r id
        let mut code = vec![0xB8 + reg];
        code.extend_from_slice(&imm.to_le_bytes());
        asm.append(&code, 1);
    }

    #[allow(dead_code)]
    fn emit_ret(&mut self, asm: &mut ElfAssembler) {
        // RET: C3
        asm.append(&[0xC3], 1);
    }

    #[allow(dead_code)]
    fn compile_return(&mut self, asm: &mut ElfAssembler, _inst: InstructionValue) -> bool {
        self.emit_ret(asm);
        true
    }
}

impl<'ctx, ASM: Assembler<LlvmIrAdaptor<'ctx>>> Backend<LlvmIrAdaptor<'ctx>, ASM> for X64Backend {
    fn gen_prologue(&mut self, _base: &mut CompilerBase<LlvmIrAdaptor<'ctx>, ASM, Self>) {
        // Simple prologue would go here
    }

    fn gen_epilogue(&mut self, _base: &mut CompilerBase<LlvmIrAdaptor<'ctx>, ASM, Self>) {
        // Simple epilogue would go here
    }

    fn compile_inst(
        &mut self,
        _base: &mut CompilerBase<LlvmIrAdaptor<'ctx>, ASM, Self>,
        inst: <LlvmIrAdaptor<'ctx> as IrAdaptor>::InstRef,
    ) -> bool {
        if let Some(inst) = inst {
            use inkwell::values::InstructionOpcode;
            match inst.get_opcode() {
                InstructionOpcode::Return => {
                    // For now, just return true without emitting
                    // Real implementation would emit return instruction
                    true
                },
                _ => false,
            }
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_llvm_adaptor_creation() {
        let context = Context::create();
        let module = context.create_module("test");
        let adaptor = LlvmIrAdaptor::new(&module);
        
        // Should have no functions initially
        assert_eq!(adaptor.funcs().count(), 0);
    }

    #[test] 
    fn test_llvm_adaptor_with_function() {
        let context = Context::create();
        let module = context.create_module("test");
        
        // Add a simple function
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let _function = module.add_function("test_func", fn_type, None);
        
        let adaptor = LlvmIrAdaptor::new(&module);
        assert_eq!(adaptor.funcs().count(), 1);
        
        // Check function name
        let func = adaptor.funcs().next().unwrap();
        assert_eq!(adaptor.func_link_name(func), "test_func");
    }
}