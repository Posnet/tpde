// This module provides all x86-64 architecture-specific functionality for the TPDE compiler.
// It contains four main components: 1) Encoder - wraps iced-x86 to emit real x86-64 machine
// code instructions with support for moves, arithmetic, comparisons, jumps, calls, and memory
// operations; 2) CallingConvention - implements System V AMD64 ABI with register assignment
// for function arguments/returns, stack frame management, and callee-saved register tracking;
// 3) FunctionCodegen - orchestrates function compilation including prologue/epilogue generation,
// stack frame setup, and integration with the encoder; 4) Backend - provides the x86-64
// implementation of the Backend trait for instruction selection. Together these components
// enable compilation from LLVM IR to native x86-64 machine code following standard ABI
// conventions. The module re-exports key types for convenient access by the compiler.

//! x86-64 architecture-specific components.
//!
//! This module contains all x86-64 specific code:
//! - Instruction encoding using iced-x86
//! - System V calling convention implementation
//! - x86-64 backend for code generation

pub mod encoder;
pub mod backend;
pub mod calling_convention;
pub mod function_codegen;

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

pub use function_codegen::{
    FunctionCodegen,
    FunctionCodegenError,
};