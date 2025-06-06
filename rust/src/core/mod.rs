//! Core TPDE Infrastructure
//!
//! This module provides the fundamental building blocks for fast IR compilation,
//! designed to be shared across different IR types and target architectures.
//! All components are optimized for compile-time performance while maintaining
//! correctness.
//!
//! # Key Components
//!
//! ## Session Management (`session`)
//! - Arena-based memory allocation using `bumpalo`
//! - Compilation statistics and performance tracking
//! - Block and value metadata storage
//!
//! ## Register Allocation (`register_file`)
//! - Clock-based register eviction algorithm
//! - Efficient register tracking with bitsets
//! - Support for both general-purpose and floating-point registers
//!
//! ## Value Management (`value_assignment`)
//! - Multi-part value representation for wide types
//! - Reference counting for efficient value tracking
//! - Storage location abstraction (register, stack, memory)
//!
//! ## Value References (`value_ref`)
//! - RAII-based value lifetime management
//! - Automatic register allocation and deallocation
//! - Type-safe value part access
//!
//! ## ELF Assembly (`assembler`)
//! - Direct ELF object file generation
//! - Symbol and relocation management
//! - Section handling for code and data
//!
//! # Design Principles
//!
//! 1. **Performance First**: Every abstraction is designed to compile away
//! 2. **Arena Allocation**: Minimize allocation overhead during compilation
//! 3. **Type Safety**: Leverage Rust's type system to prevent bugs
//! 4. **Zero-Cost Abstractions**: No runtime overhead for safety features

pub mod session;
pub mod register_file;
pub mod value_assignment;
pub mod error;
pub mod assembler;
pub mod value_ref;
pub mod adaptor;
pub mod analyzer;
pub mod compiler;

// Re-export core components
pub use session::{
    CompilationSession,
    SessionStats,
    SessionError,
    ValueLocation,
    BlockInfo,
    PhiNodeInfo,
};

pub use register_file::{
    RegisterFile,
    AsmReg,
    RegBitSet,
    RegAllocError,
};

pub use value_assignment::{
    ValueAssignment,
    ValueAssignmentManager,
    ValLocalIdx,
    AssignmentFlags,
    AssignmentAllocator,
    StorageLocation,
};

pub use error::{
    CompileError,
    CompileResult,
};

pub use assembler::{
    Assembler,
    ElfAssembler,
};

pub use value_ref::{
    ValueRef,
    ValuePartRef,
    ValueRefBuilder,
    ValueRefError,
    CompilerContext,
};

pub use adaptor::IrAdaptor;
pub use analyzer::Analyzer;
pub use compiler::{Backend, CompilerBase};