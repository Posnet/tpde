//! Core infrastructure shared across all compilation targets.
//!
//! This module contains the fundamental components used by all backends:
//! - Session management with arena allocation
//! - Register allocation and tracking
//! - Value assignment and storage management
//! - Error types
//! - ELF assembly generation

// Re-export core components (temporary during migration)
pub use crate::compilation_session::{
    CompilationSession,
    SessionStats,
    SessionError,
    ValueLocation,
    BlockInfo,
    PhiNodeInfo,
};

pub use crate::register_file::{
    RegisterFile,
    AsmReg,
    RegBitSet,
    RegAllocError,
};

pub use crate::value_assignment::{
    ValueAssignment,
    ValueAssignmentManager,
    ValLocalIdx,
    AssignmentFlags,
    AssignmentAllocator,
    StorageLocation,
};

pub use crate::error::{
    CompileError,
    CompileResult,
};

pub use crate::assembler::{
    Assembler,
    ElfAssembler,
};

// Future modules (to be moved here):
// pub mod session;           // From compilation_session.rs
// pub mod register_file;     // From register_file.rs
// pub mod value_assignment;  // From value_assignment.rs
// pub mod error;            // From error.rs
// pub mod assembler;        // From assembler.rs