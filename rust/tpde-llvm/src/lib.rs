//! TPDE LLVM backend in Rust.
//!
//! This crate reimplements the C++ TPDE-LLVM front-end.  It targets x86-64 and
//! AArch64 ELF and strives for compile times around 10--20x faster than LLVM's
//! `-O0` back-end while producing comparable baseline code quality.  Only a
//! commonly used subset of LLVM-IR is supported; unsupported features should be
//! handled by falling back to LLVM.  A condensed description of the original
//! design and current limitations lives in [`tpde_core::overview`].

use inkwell::module::Module;
use tpde_core::{adaptor::IrAdaptor, assembler::Assembler, compiler::CompilerBase};

/// Compile an LLVM `Module` using a TPDE compiler setup.
pub fn compile_ir<A, ASM>(
    _module: &Module,
    _adaptor: A,
    _assembler: ASM,
) -> CompilerBase<A, ASM>
where
    A: IrAdaptor,
    ASM: Assembler<A>,
{
    unimplemented!("LLVM compilation not yet implemented")
}

/// Simple text marker proving the crate works.
pub fn compiler() -> &'static str {
    "TPDE LLVM backend"
}
