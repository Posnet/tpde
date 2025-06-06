//! Legacy implementations kept for reference.
//!
//! This module contains older implementations that have been superseded
//! by the direct LLVM integration approach. These are kept for:
//! - Historical reference
//! - Test compatibility
//! - Understanding the evolution of the design
//!
//! ⚠️ **Warning**: Code in this module is deprecated and should not be
//! used for new development. Use `tpde::llvm` instead.

#![deprecated(
    since = "0.2.0",
    note = "Legacy code - use tpde::llvm for production"
)]

pub mod generic_compiler;
pub mod minimal_adaptor;

// Re-export old generic compiler with clearer name
#[deprecated(note = "Use llvm::LlvmCompiler instead")]
pub use generic_compiler::{
    CompleteCompiler as GenericCompiler,
    AddressingMode,
    GepExpression,
    CompilerError as GenericCompileError,
};

// Re-export minimal adaptor (was basic.rs)
#[deprecated(note = "Use llvm::LlvmAdaptor instead")]
pub use minimal_adaptor::{
    LlvmIrAdaptor as MinimalAdaptor,
};