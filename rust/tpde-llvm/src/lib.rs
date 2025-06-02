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
};
use tpde_core::{adaptor::IrAdaptor, assembler::Assembler, compiler::{CompilerBase, Backend}};

/// Adaptor walking an LLVM [`Module`] using `inkwell`.
///
/// This very small implementation only exposes functions which is enough for
/// [`CompilerBase::compile`].  Values, blocks and instructions are represented
/// as `Option` types so we can define an invalid constant.
pub struct LlvmIrAdaptor<'ctx> {
    module: Module<'ctx>,
    funcs: Vec<FunctionValue<'ctx>>,
    names: Vec<String>,
    current: Option<FunctionValue<'ctx>>,
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
            module: module.clone(),
            funcs,
            names,
            current: None,
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
        func.is_some()
    }

    fn reset(&mut self) {
        self.current = None;
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
        let _ = inst;
        Box::new(std::iter::empty())
    }

    fn inst_results(&self, _inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        Box::new(std::iter::empty())
    }

    fn val_local_idx(&self, _val: Self::ValueRef) -> usize {
        0
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

/// Build a [`CompilerBase`] ready to process the LLVM `Module`.
///
/// The returned compiler can immediately run [`CompilerBase::compile`].  The
/// current implementation only gathers functions; instruction selection is
/// still a stub.
pub fn compile_ir<'ctx, B>(module: &Module<'ctx>, backend: B) -> CompilerBase<LlvmIrAdaptor<'ctx>, NullAssembler, B>
where
    B: Backend<LlvmIrAdaptor<'ctx>, NullAssembler>,
{
    let adaptor = LlvmIrAdaptor::new(module);
    let assembler = NullAssembler::new(false);
    CompilerBase::new(adaptor, assembler, backend)
}

/// Simple text marker proving the crate works.
pub fn compiler() -> &'static str {
    "TPDE LLVM backend"
}
