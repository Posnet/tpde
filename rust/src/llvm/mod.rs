//! Primary LLVM compilation implementation.
//!
//! This module contains the main LLVM IR compiler for TPDE. It provides
//! direct LLVM integration without generic trait bounds, following the
//! Phase 2B architectural redesign.
//!
//! # Example
//! ```ignore
//! use tpde::llvm::{LlvmCompiler, LlvmAdaptor};
//! use tpde::core::CompilationSession;
//! use bumpalo::Bump;
//! 
//! let arena = Bump::new();
//! let session = CompilationSession::new(&arena);
//! let mut compiler = LlvmCompiler::new(&module, &session)?;
//! compiler.compile_function("main")?;
//! ```

// Temporary re-export during migration
pub use crate::llvm_compiler_concrete::{
    LlvmCompiler,
    CompiledFunction,
    LlvmCompilerError,
};

// Re-export the production adaptor (was enhanced.rs)
pub use crate::llvm_adaptor::enhanced::{
    EnhancedLlvmAdaptor as LlvmAdaptor,
    PhiInfo,
};

// Re-export instruction categories
pub use crate::llvm_compiler::{
    InstructionCategory,
    LlvmAdaptorInterface,
};

// Future modules (to be moved here):
// pub mod adaptor;           // From llvm_adaptor/enhanced.rs
// pub mod instruction_select; // Extract from llvm_compiler_concrete.rs
// pub mod phi_resolution;     // From phi_resolver.rs
// pub mod function_analysis;  // From function_analyzer_arena.rs
// pub mod gep;               // Extract GEP-related code