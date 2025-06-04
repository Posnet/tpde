//! Core TPDE framework rewritten in Rust.
//!
//! This crate contains the foundation of TPDE.  It mirrors the
//! architecture laid out in the original C++ implementation while adopting
//! Rust idioms.  The framework aims for extremely fast compilation of SSA based
//! IRs by providing just enough infrastructure for register allocation, stack
//! management, unwind info and ELF emission.  Instruction selection is left to
//! the user, either handwritten or stitched together from snippet encoders.
//!
//! High level pieces are:
//! - [`overview`] summarises the overall framework design.
//! - [`adaptor`] defines the [`IrAdaptor`] trait implemented by an IR frontend.
//! - [`analyzer`] computes block order and liveness.
//! - [`compiler`] implements the architecture neutral driver.
//! - [`assembler`] is the trait for emitting object files.

pub mod overview;
pub mod guide;
pub mod adaptor;
pub mod analyzer;
pub mod assembler;
pub mod compiler;
pub mod value_assignment;
pub mod register_file;
pub mod value_ref;
pub mod x64_encoder;
pub mod x64_backend;
pub mod calling_convention;
pub mod function_codegen;
pub use compiler::{Backend, CompilerBase};
pub use value_assignment::{ValueAssignment, ValueAssignmentManager, ValLocalIdx};
pub use register_file::{RegisterFile, AsmReg, RegBitSet, RegAllocError};
pub use value_ref::{ValueRef, ValuePartRef, ValueRefBuilder, ValueRefError, CompilerContext};

/// Temporary hello world to prove the crate builds.
pub fn hello() -> &'static str {
    "Hello from tpde"
}
