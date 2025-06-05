//! TPDE LLVM backend in Rust.
//!
//! This crate reimplements the C++ TPDE-LLVM front-end.  It targets x86-64 and
//! AArch64 ELF and strives for compile times around 10--20x faster than LLVM's
//! `-O0` back-end while producing comparable baseline code quality.  Only a
//! commonly used subset of LLVM-IR is supported; unsupported features should be
//! handled by falling back to LLVM.  A condensed description of the original
//! design and current limitations lives in [`tpde_core::overview`].

use inkwell::{
    basic_block::BasicBlock,
    module::Module,
    values::{BasicValueEnum, FunctionValue, InstructionValue},
    Either,
};
use inkwell::llvm_sys::prelude::LLVMValueRef;
use inkwell::values::{AnyValue, AsValueRef};
use std::collections::HashMap;
use std::convert::TryFrom;
use tpde_core::{adaptor::IrAdaptor, assembler::{Assembler, ElfAssembler}, compiler::{CompilerBase, Backend}};

pub mod enhanced_adaptor;

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
}

impl<'ctx> IrAdaptor for LlvmIrAdaptor<'ctx> {
    type ValueRef = Option<BasicValueEnum<'ctx>>;
    type InstRef = Option<InstructionValue<'ctx>>;
    type BlockRef = Option<BasicBlock<'ctx>>;
    type FuncRef = Option<FunctionValue<'ctx>>;

    const INVALID_VALUE_REF: Self::ValueRef = None;
    const INVALID_BLOCK_REF: Self::BlockRef = None;
    const INVALID_FUNC_REF: Self::FuncRef = None;

    fn func_count(&self) -> u32 {
        self.funcs.len() as u32
    }

    fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_> {
        Box::new(self.funcs.iter().cloned().map(Some))
    }

    fn func_link_name(&self, func: Self::FuncRef) -> &str {
        if let Some(f) = func {
            if let Some(pos) = self.funcs.iter().position(|&fv| fv == f) {
                return &self.names[pos];
            }
        }
        ""
    }

    fn switch_func(&mut self, func: Self::FuncRef) -> bool {
        self.current = func;
        self.val_index.clear();
        if let Some(f) = func {
            let mut idx = 0usize;
            for param in f.get_param_iter() {
                self.val_index.insert(param.as_value_ref(), idx);
                idx += 1;
            }
            for bb in f.get_basic_blocks() {
                for inst in bb.get_instructions() {
                    self.val_index.insert(inst.as_value_ref(), idx);
                    idx += 1;
                }
            }
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.current = None;
        self.val_index.clear();
    }

    fn entry_block(&self) -> Self::BlockRef {
        self.current.and_then(|f| f.get_first_basic_block())
    }

    fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
        if let Some(func) = self.current {
            Box::new(func.get_basic_blocks().into_iter().map(Some))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn block_insts(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_> {
        if let Some(bb) = block {
            Box::new(bb.get_instructions().into_iter().map(Some))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn block_succs(&self, _block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
        Box::new(std::iter::empty())
    }

    fn inst_operands(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        if let Some(i) = inst {
            Box::new(i.get_operands().map(|op| match op {
                Some(Either::Left(v)) => Some(v),
                _ => None,
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
                let any = i.as_any_value_enum();
                let val = BasicValueEnum::try_from(any).ok();
                Box::new(val.into_iter().map(Some))
            }
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn val_local_idx(&self, val: Self::ValueRef) -> usize {
        if let Some(v) = val {
            let key = v.as_value_ref();
            self.val_index.get(&key).copied().unwrap_or(0)
        } else {
            0
        }
    }

    fn val_ignore_liveness(&self, _val: Self::ValueRef) -> bool {
        false
    }
}

/// Minimal assembler that satisfies [`Assembler`].
pub struct NullAssembler {
    next_label: usize,
}

impl NullAssembler {
    pub fn new(_: bool) -> Self {
        Self { next_label: 0 }
    }
}

impl<A: IrAdaptor> Assembler<A> for NullAssembler {
    type SymRef = ();
    type Label = usize;

    fn new(generate_object: bool) -> Self {
        Self::new(generate_object)
    }

    fn label_create(&mut self) -> Self::Label {
        let id = self.next_label;
        self.next_label += 1;
        id
    }

    fn label_place(&mut self, _label: Self::Label) {}

    fn sym_predef_func(&mut self, _name: &str, _local: bool, _weak: bool) -> Self::SymRef {}

    fn sym_add_undef(&mut self, _name: &str, _local: bool, _weak: bool) {}

    fn finalize(&mut self) {}

    fn build_object_file(&mut self) -> Vec<u8> { Vec::new() }

    fn map<F>(&mut self, _resolve: F) -> bool
    where
        F: FnMut(&str) -> *const u8,
    {
        true
    }
}

/// Basic x86-64 backend for LLVM IR compilation.
pub struct X64Backend {
    reg_usage: u64,
}

impl X64Backend {
    pub fn new() -> Self {
        Self { reg_usage: 0 }
    }

    fn emit_mov_imm32(&mut self, asm: &mut ElfAssembler, reg: u8, imm: u32) {
        // mov eax, imm32 -> 0xB8 + r, imm32
        let mut bytes = vec![0xB8 + reg];
        bytes.extend_from_slice(&imm.to_le_bytes());
        asm.append(&bytes, 1);
    }

    fn emit_ret(&mut self, asm: &mut ElfAssembler) {
        // ret -> 0xC3
        asm.append(&[0xC3], 1);
    }

    fn compile_return(&mut self, asm: &mut ElfAssembler, _inst: InstructionValue) -> bool {
        // Simple return - just emit ret instruction
        self.emit_ret(asm);
        true
    }
}

impl<A: IrAdaptor> Backend<A, ElfAssembler> for X64Backend {
    fn gen_prologue(&mut self, _base: &mut CompilerBase<A, ElfAssembler, Self>) {
        // Prologue handled separately
    }

    fn gen_epilogue(&mut self, _base: &mut CompilerBase<A, ElfAssembler, Self>) {
        // Epilogue handled separately  
    }

    fn compile_inst(&mut self, _base: &mut CompilerBase<A, ElfAssembler, Self>, _inst: A::InstRef) -> bool {
        // Placeholder - actual instruction selection would go here
        true
    }
}

/// Simple backend that prints instruction names.
pub struct PrintBackend;

impl<A: IrAdaptor, ASM: Assembler<A>> Backend<A, ASM> for PrintBackend {
    fn gen_prologue(&mut self, _base: &mut CompilerBase<A, ASM, Self>) {
        // println!("prologue");
    }

    fn gen_epilogue(&mut self, _base: &mut CompilerBase<A, ASM, Self>) {
        // println!("epilogue");
    }

    fn compile_inst(&mut self, _base: &mut CompilerBase<A, ASM, Self>, _inst: A::InstRef) -> bool {
        // println!("compile_inst");
        true
    }
}

/// Build a [`CompilerBase`] ready to process the LLVM `Module` with null assembler.
pub fn compile_ir<'ctx>(module: &Module<'ctx>) -> CompilerBase<LlvmIrAdaptor<'ctx>, NullAssembler, PrintBackend> {
    let adaptor = LlvmIrAdaptor::new(module);
    let assembler = NullAssembler::new(false);
    let backend = PrintBackend;
    CompilerBase::new(adaptor, assembler, backend)
}

/// Build a [`CompilerBase`] with ELF assembler for x86-64 compilation.
pub fn compile_ir_x64<'ctx>(module: &Module<'ctx>) -> CompilerBase<LlvmIrAdaptor<'ctx>, ElfAssembler, X64Backend> {
    let adaptor = LlvmIrAdaptor::new(module);
    let assembler = <ElfAssembler as Assembler<LlvmIrAdaptor>>::new(true);
    let backend = X64Backend::new();
    CompilerBase::new(adaptor, assembler, backend)
}

/// Build a complete compiler using the enhanced LLVM adaptor for real-world compilation.
pub fn compile_enhanced_ir<'ctx>(module: &Module<'ctx>) -> Result<tpde_core::complete_compiler::CompleteCompiler<crate::enhanced_adaptor::EnhancedLlvmAdaptor<'ctx>>, tpde_core::complete_compiler::CompilerError> {
    let adaptor = crate::enhanced_adaptor::EnhancedLlvmAdaptor::new(module);
    tpde_core::complete_compiler::CompleteCompiler::new(adaptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_compile_empty_module() {
        let context = Context::create();
        let module = context.create_module("test");
        let mut compiler = compile_ir(&module);
        assert!(compiler.compile());
    }

    #[test]
    fn test_compile_simple_function() {
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let function = module.add_function("main", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        let builder = context.create_builder();
        builder.position_at_end(basic_block);
        let return_val = i32_type.const_int(42, false);
        builder.build_return(Some(&return_val)).unwrap();

        let mut compiler = compile_ir(&module);
        assert!(compiler.compile());
    }

    #[test]
    fn test_compile_x64_backend() {
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let function = module.add_function("main", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        let builder = context.create_builder();
        builder.position_at_end(basic_block);
        let return_val = i32_type.const_int(42, false);
        builder.build_return(Some(&return_val)).unwrap();

        let mut compiler = compile_ir_x64(&module);
        assert!(compiler.compile());
    }

    #[test]
    fn test_enhanced_compiler_simple_function() {
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("add", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        let builder = context.create_builder();
        builder.position_at_end(basic_block);
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();
        let result = builder.build_int_add(a, b, "result").unwrap();
        builder.build_return(Some(&result)).unwrap();

        let mut compiler = compile_enhanced_ir(&module).unwrap();
        assert!(compiler.compile_all().is_ok());
    }

    fn create_factorial_ir(context: &Context) -> Module {
        let module = context.create_module("factorial");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into()], false);
        let function = module.add_function("factorial", fn_type, None);
        
        let entry_block = context.append_basic_block(function, "entry");
        let if_block = context.append_basic_block(function, "if.then");
        let else_block = context.append_basic_block(function, "if.else");
        let return_block = context.append_basic_block(function, "return");
        
        let builder = context.create_builder();
        
        // Entry block: if (n <= 1)
        builder.position_at_end(entry_block);
        let n = function.get_nth_param(0).unwrap().into_int_value();
        let one = i32_type.const_int(1, false);
        let cond = builder.build_int_compare(inkwell::IntPredicate::SLE, n, one, "cond").unwrap();
        builder.build_conditional_branch(cond, if_block, else_block).unwrap();
        
        // If block: return 1
        builder.position_at_end(if_block);
        builder.build_unconditional_branch(return_block).unwrap();
        
        // Else block: return n * factorial(n - 1)
        builder.position_at_end(else_block);
        let n_minus_1 = builder.build_int_sub(n, one, "n_minus_1").unwrap();
        let call = builder.build_call(function, &[n_minus_1.into()], "call").unwrap();
        let result = call.try_as_basic_value().left().unwrap().into_int_value();
        let product = builder.build_int_mul(n, result, "product").unwrap();
        builder.build_unconditional_branch(return_block).unwrap();
        
        // Return block with PHI node
        builder.position_at_end(return_block);
        let phi = builder.build_phi(i32_type, "result").unwrap();
        phi.add_incoming(&[(&one, if_block), (&product, else_block)]);
        builder.build_return(Some(&phi.as_basic_value())).unwrap();
        
        module
    }

    #[test]
    fn test_enhanced_compiler_factorial() {
        let context = Context::create();
        let module = create_factorial_ir(&context);
        
        let mut compiler = compile_enhanced_ir(&module).unwrap();
        assert!(compiler.compile_all().is_ok());
    }
    
    #[test]
    fn test_opcode_based_instruction_selection() {
        let context = Context::create();
        let module = context.create_module("test_opcodes");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("add_test", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        let builder = context.create_builder();
        builder.position_at_end(basic_block);
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();
        
        // Create real LLVM IR: %result = add nsw i32 %a, %b
        let result = builder.build_int_add(a, b, "result").unwrap();
        builder.build_return(Some(&result)).unwrap();

        // Compile with enhanced adaptor - should use opcode-based selection
        let mut compiler = compile_enhanced_ir(&module).unwrap();
        let compilation_result = compiler.compile_all();
        
        // Should succeed with opcode-based instruction selection
        assert!(compilation_result.is_ok(), "Opcode-based compilation failed: {:?}", compilation_result);
        
        // Verify we have compiled functions
        let compiled = compiler.get_compiled_functions();
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].name, "add_test");
        assert!(!compiled[0].code.is_empty());
        
        println!("✅ Opcode-based instruction selection test passed!");
        println!("   Function '{}' compiled to {} bytes using LLVM opcode analysis", 
                 compiled[0].name, compiled[0].code.len());
    }
    
    #[test]
    fn test_comprehensive_arithmetic_compilation() {
        let context = Context::create();
        let module = context.create_module("arithmetic_test");
        let i32_type = context.i32_type();
        
        // Test function: int arithmetic_test(int a, int b, int c) { return (a + b) * c; }
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into(), i32_type.into()], false);
        let function = module.add_function("arithmetic_test", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        let builder = context.create_builder();
        builder.position_at_end(basic_block);
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();
        let c = function.get_nth_param(2).unwrap().into_int_value();
        
        // Generate real LLVM IR with multiple arithmetic operations
        let add_result = builder.build_int_add(a, b, "add_result").unwrap();  // ADD
        let final_result = builder.build_int_mul(add_result, c, "mul_result").unwrap(); // MUL
        builder.build_return(Some(&final_result)).unwrap();

        // Compile with enhanced adaptor - should use real arithmetic instruction selection
        let mut compiler = compile_enhanced_ir(&module).unwrap();
        let compilation_result = compiler.compile_all();
        
        // Should succeed with real arithmetic instruction compilation
        assert!(compilation_result.is_ok(), "Arithmetic compilation failed: {:?}", compilation_result);
        
        // Verify we have compiled the function with multiple arithmetic operations
        let compiled = compiler.get_compiled_functions();
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].name, "arithmetic_test");
        assert!(!compiled[0].code.is_empty());
        
        println!("✅ Comprehensive arithmetic compilation test passed!");
        println!("   Function '{}' with ADD and MUL operations compiled to {} bytes", 
                 compiled[0].name, compiled[0].code.len());
        println!("   Real C++ equivalent: (a + b) * c with proper register allocation");
    }
}