// This module implements direct LLVM IR compilation with a concrete design that eliminates
// abstraction overhead after the architectural redesign. It provides the LlvmCompiler for
// function-to-machine-code compilation, LlvmAdaptor for IR traversal and analysis, and
// supporting infrastructure for instruction selection. The implementation uses direct
// InstructionValue manipulation with opcode-based dispatch, avoiding trait bounds for
// clean LLVM API access. Supports arithmetic (add/sub/mul), comparisons (icmp), memory
// operations (load/store/alloca/GEP), basic control flow (unconditional br, ret), and
// direct function calls. Major gaps include PHI nodes (preventing loops), conditional
// branches, switch statements, exception handling, and intrinsics. Uses arena allocation
// via CompilationSession for efficient memory management and integrates tightly with
// inkwell for LLVM bindings. This concrete approach replaced the generic framework after
// hitting trait bound complexity issues during enhanced adaptor integration.

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
pub mod analysis_result;
pub mod call_support;
pub mod function_analysis;
pub mod traits;

// Main exports
pub use adaptor::{EnhancedLlvmAdaptor as LlvmAdaptor, PhiInfo};
pub use compiler::{CompiledFunction, LlvmCompiler, LlvmCompilerError};
pub use traits::{InstructionCategory, LlvmAdaptorInterface};
