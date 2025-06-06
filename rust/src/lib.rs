//! TPDE Rust Implementation - Fast LLVM IR Compilation
//!
//! TPDE (Turbo-Powered Direct Execution) provides extremely fast compilation 
//! of LLVM IR with a focus on low-latency code generation. It compiles 10-20x 
//! faster than LLVM -O0 while maintaining reasonable code quality.
//!
//! # Status: ~70-75% Complete
//!
//! The Rust implementation has completed a major architectural redesign that
//! provides direct LLVM integration without abstraction overhead. Key features
//! include arena-based memory management, concrete LLVM compiler implementation,
//! and comprehensive x86-64 instruction support.
//!
//! # Primary Usage
//! 
//! ```ignore
//! use tpde::llvm::LlvmCompiler;
//! use tpde::core::CompilationSession;
//! use bumpalo::Bump;
//! use inkwell::context::Context;
//! 
//! // Create LLVM context and module
//! let context = Context::create();
//! let module = context.create_module("example");
//! 
//! // Create compilation session with arena allocation
//! let arena = Bump::new();
//! let session = CompilationSession::new(&arena);
//! 
//! // Compile LLVM IR module directly
//! let mut compiler = LlvmCompiler::new(&module, &session);
//! let func = module.get_function("main").unwrap();
//! let result = compiler.compile_function(func)?;
//! ```
//!
//! # Architecture
//! 
//! The implementation uses a concrete, LLVM-specific design that eliminates
//! trait bound complexity:
//!
//! - [`llvm`] - Direct LLVM compiler implementation with opcode dispatch
//! - [`core`] - Core infrastructure (values, registers, sessions)
//! - [`x64`] - x86-64 backend with instruction encoding and ABI
//! - [`experimental`] - Future features (ARM64 support planned)
//!
//! # Key Design Decisions
//!
//! 1. **Arena Allocation**: All compilation objects are allocated in arenas
//!    tied to the compilation session lifetime
//! 2. **Direct LLVM Integration**: No abstraction layers - direct use of
//!    inkwell's LLVM bindings for maximum performance
//! 3. **Concrete Types**: Eliminated generic traits in favor of concrete
//!    LLVM-specific types for cleaner code
//!
//! # Completed Features
//!
//! - ✅ Arithmetic operations (add, sub, mul)
//! - ✅ Comparisons (icmp with all predicates)
//! - ✅ Memory operations (load, store, alloca, GEP)
//! - ✅ Function calls (direct calls with System V ABI)
//! - ✅ Basic control flow (unconditional branches, returns)
//! - ✅ x86-64 code generation with proper register allocation
//!
//! # Missing Features
//!
//! - ❌ PHI nodes (critical for loops)
//! - ❌ Conditional branches and switch statements
//! - ❌ Exception handling (invoke/landingpad)
//! - ❌ LLVM intrinsics
//! - ❌ ARM64 backend
//!
//! # Documentation
//!
//! For detailed information, see:
//! - `rust/README.md` - Rust implementation overview
//! - `CLAUDE.md` - Comprehensive project documentation
//! - `docs/tpde/` - Framework architecture and design

// Main modules
pub mod llvm;
pub mod core;
pub mod x64;
pub mod experimental;

// Utility modules
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

