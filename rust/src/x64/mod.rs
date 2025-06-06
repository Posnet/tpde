//! x86-64 architecture-specific components.
//!
//! This module contains all x86-64 specific code:
//! - Instruction encoding using iced-x86
//! - System V calling convention implementation
//! - x86-64 backend for code generation

pub mod encoder;
pub mod backend;
pub mod calling_convention;

// Re-export x64 components
pub use encoder::{
    X64Encoder as Encoder,
    EncodingError,
    JumpCondition,
    InstructionSelector,
};

pub use backend::{
    X64Backend as Backend,
};

pub use calling_convention::{
    SysVAssigner as CallingConvention,
    CCAssigner,
    CCAssignment,
    RegBank,
    FunctionFrame,
};