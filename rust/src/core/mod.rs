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