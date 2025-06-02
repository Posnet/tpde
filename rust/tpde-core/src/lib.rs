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

/// Temporary hello world to prove the crate builds.
pub fn hello() -> &'static str {
    "Hello from tpde"
}
