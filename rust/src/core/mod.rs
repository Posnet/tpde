// This module serves as the central hub for TPDE's core infrastructure components,
// providing fundamental building blocks for fast IR compilation that are shared across
// different IR types and target architectures. It exports and organizes key subsystems:
// session management (arena-based memory allocation and compilation statistics),
// register allocation (clock-based eviction with bitset tracking), value management
// (multi-part values with reference counting and storage locations), value references
// (RAII-based lifetime management), ELF assembly (direct object file generation),
// and framework traits (IrAdaptor, Analyzer, Backend, CompilerBase). All components
// are designed with performance as the primary goal, using arena allocation to minimize
// overhead, zero-cost abstractions, and type safety to prevent bugs at compile time.

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

pub mod adaptor;
pub mod analyzer;
pub mod assembler;
pub mod compiler;
pub mod error;
pub mod register_file;
pub mod session;
pub mod value_assignment;
pub mod value_ref;

#[cfg(test)]
pub mod test_utils;

// Re-export core components
pub use session::{
    BlockInfo, CompilationMetrics, CompilationSession, PhiNodeInfo, SessionError, SessionStats,
    ValueLocation,
};

pub use register_file::{AsmReg, RegAllocError, RegBitSet, RegisterFile};

pub use value_assignment::{
    AssignmentAllocator, AssignmentFlags, StorageLocation, ValLocalIdx, ValueAssignment,
    ValueAssignmentManager,
};

pub use error::{CompileError, CompileResult};

pub use assembler::{Assembler, ElfAssembler};

pub use value_ref::{CompilerContext, ValuePartRef, ValueRef, ValueRefBuilder, ValueRefError};

pub use adaptor::IrAdaptor;
pub use analyzer::Analyzer;
pub use compiler::{Backend, CompilerBase};
