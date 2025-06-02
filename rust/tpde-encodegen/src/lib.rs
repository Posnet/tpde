//! Encoding snippet generator entry points.
//!
//! `tpde-encodegen` transforms short high level functions (typically written in
//! C and compiled to LLVM IR) into Rust routines that emit the matching machine
//! instructions. The generated code reuses registers when possible and can fuse
//! constant operands or address expressions. By mixing these snippet encoders
//! into a compiler one can implement most instruction selection logic without
//! writing every pattern by hand. See [`tpde_core::overview`] for an
//! extended overview.

use inkwell::module::Module;

/// Parse the provided LLVM IR module and emit snippet encoders.
#[allow(dead_code)]
pub fn generate(_module: &Module) {
    todo!("encode generation not yet implemented")
}
