//! TPDE LLVM backend in Rust.

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
