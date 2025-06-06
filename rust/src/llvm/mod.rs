//! LLVM IR Compiler Implementation
//!
//! This module provides direct LLVM IR compilation with a concrete, optimized
//! implementation that eliminates abstraction overhead. The design focuses on
//! compile-time performance while maintaining correctness.
//!
//! # Architecture
//!
//! The LLVM compiler uses a concrete design pattern:
//! - Direct `InstructionValue<'ctx>` manipulation
//! - Opcode-based dispatch without trait bounds
//! - Arena allocation for compilation objects
//! - Tight integration with inkwell LLVM bindings
//!
//! # Key Components
//!
//! ## `LlvmCompiler` - Main Compiler
//! The primary compiler struct that orchestrates:
//! - Function compilation from LLVM IR to machine code
//! - Register allocation and value management
//! - Instruction selection and encoding
//! - ELF object file generation
//!
//! ## `LlvmAdaptor` - IR Navigation
//! Provides efficient traversal of LLVM IR:
//! - Block iteration in reverse post-order
//! - Instruction categorization by opcode
//! - PHI node analysis (placeholder)
//! - Value and type introspection
//!
//! ## Function Analysis
//! - Control flow graph analysis
//! - Block successor/predecessor relationships
//! - Live value tracking
//! - PHI node dependency analysis (planned)
//!
//! # Supported Instructions
//!
//! ✅ **Fully Implemented:**
//! - Arithmetic: add, sub, mul
//! - Comparison: icmp (all predicates)
//! - Memory: load, store, alloca, getelementptr
//! - Control: br (unconditional), ret
//! - Calls: direct function calls
//!
//! ❌ **Not Yet Implemented:**
//! - PHI nodes (critical gap)
//! - Conditional branches
//! - Switch statements
//! - Exception handling (invoke/landingpad)
//! - LLVM intrinsics
//!
//! # Example
//! ```ignore
//! use tpde::llvm::LlvmCompiler;
//! use tpde::core::CompilationSession;
//! use bumpalo::Bump;
//! use inkwell::context::Context;
//! 
//! let context = Context::create();
//! let module = context.create_module("example");
//! // ... populate module with LLVM IR ...
//! 
//! let arena = Bump::new();
//! let session = CompilationSession::new(&arena);
//! let mut compiler = LlvmCompiler::new(&module, &session);
//! 
//! let func = module.get_function("main").unwrap();
//! let compiled = compiler.compile_function(func)?;
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