//! x86-64 architecture-specific components.
//!
//! This module contains all x86-64 specific code:
//! - Instruction encoding using iced-x86
//! - System V calling convention implementation
//! - x86-64 backend for code generation

// Re-export x64 components (temporary during migration)
pub use crate::x64_encoder::{
    X64Encoder as Encoder,
    EncodingError,
    JumpCondition,
    InstructionSelector,
};

pub use crate::x64_backend::{
    X64Backend as Backend,
};

pub use crate::calling_convention::{
    SysVAssigner as CallingConvention,
    CCAssigner,
    CCAssignment,
    RegBank,
    FunctionFrame,
};

// Future modules (to be moved here):
// pub mod encoder;           // From x64_encoder.rs
// pub mod backend;           // From x64_backend.rs
// pub mod calling_convention; // From calling_convention.rs (System V part)
// pub mod addressing;        // Extract AddressingMode and related code