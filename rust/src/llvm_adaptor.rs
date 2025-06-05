//! LLVM IR adaptor module.
//!
//! This module contains the LLVM-specific adaptors and functionality
//! that was previously in the tpde-llvm crate.

pub mod basic;
pub mod enhanced;

// Re-export the main types
pub use basic::LlvmIrAdaptor;
pub use enhanced::EnhancedLlvmAdaptor;