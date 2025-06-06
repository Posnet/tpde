//! TPDE - Fast SSA-based IR Compilation.
//!
//! TPDE (Two Phase Deoptimization Engine) provides extremely fast compilation 
//! of SSA-based IRs with a focus on LLVM IR. It compiles 10-20x faster than 
//! LLVM -O0 while maintaining reasonable code quality.
//!
//! # Primary Usage
//! 
//! ```ignore
//! use tpde::llvm::{LlvmCompiler, LlvmAdaptor};
//! use tpde::core::CompilationSession;
//! use bumpalo::Bump;
//! 
//! // Create compilation session with arena allocation
//! let arena = Bump::new();
//! let session = CompilationSession::new(&arena);
//! 
//! // Compile LLVM IR module
//! let mut compiler = LlvmCompiler::new(&module, &session)?;
//! let result = compiler.compile_function("main")?;
//! ```
//!
//! # Architecture
//! 
//! - [`llvm`] - Primary LLVM compilation implementation
//! - [`core`] - Shared infrastructure (session, registers, values)
//! - [`x64`] - x86-64 specific code (encoder, calling convention)
//! - [`legacy`] - Deprecated implementations for reference
//! - [`experimental`] - Work in progress features

// New organized modules
pub mod llvm;
pub mod core;
pub mod x64;
pub mod legacy;
pub mod experimental;

// Other modules
pub mod docs;
pub mod encodegen;

// Re-export common types from organized modules
pub use core::{
    // Framework traits
    Backend, CompilerBase, IrAdaptor, Analyzer,
    // Value management
    ValueAssignment, ValueAssignmentManager, ValLocalIdx,
    ValueRef, ValuePartRef, ValueRefBuilder, ValueRefError, CompilerContext,
    // Register allocation
    RegisterFile, AsmReg, RegBitSet, RegAllocError,
    // Session management
    CompilationSession, SessionStats, SessionError,
};
pub use llvm::{LlvmCompiler as LlvmCompilerConcrete, CompiledFunction, LlvmCompilerError};

// =============================================================================
// Compatibility Layer - Deprecated Aliases
// =============================================================================

/// Primary LLVM compiler (was llvm_compiler_concrete::LlvmCompiler).
#[deprecated(since = "0.2.0", note = "Use llvm::LlvmCompiler directly")]
pub use self::llvm::LlvmCompiler;

/// Generic trait-based compiler (legacy).
#[deprecated(since = "0.2.0", note = "Use llvm::LlvmCompiler instead of generic approach")]
pub use self::legacy::GenericCompiler as CompleteCompiler;

/// Temporary hello world to prove the crate builds.
pub fn hello() -> &'static str {
    "Hello from tpde"
}
