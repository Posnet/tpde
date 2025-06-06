//! Core infrastructure shared across all compilation targets.
//!
//! This module contains the fundamental components used by all backends:
//! - Session management with arena allocation
//! - Register allocation and tracking
//! - Value assignment and storage management
//! - Error types
//! - ELF assembly generation

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