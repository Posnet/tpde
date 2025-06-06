//! Primary LLVM compilation implementation.
//!
//! This module contains the main LLVM IR compiler for TPDE. It provides
//! direct LLVM integration without generic trait bounds.
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

pub mod adaptor;
pub mod compiler;
// pub mod phi_resolution; // Not yet implemented - needs FunctionAnalyzer API updates
pub mod function_analysis;
pub mod analysis_result;
pub mod traits;

// Main exports
pub use compiler::{LlvmCompiler, CompiledFunction, LlvmCompilerError};
pub use adaptor::{EnhancedLlvmAdaptor as LlvmAdaptor, PhiInfo};
pub use traits::{InstructionCategory, LlvmAdaptorInterface};