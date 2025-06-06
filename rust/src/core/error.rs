// This module defines error types for the TPDE compiler using the thiserror crate for
// idiomatic Rust error handling. CompileError is the main error enum covering various
// failure scenarios: unsupported instruction widths/opcodes, register allocation failures,
// code generation errors, missing functions, invalid LLVM values, PHI node cycles, and
// block layout issues. Each variant carries relevant context (operation names, widths,
// opcodes, error reasons) for debugging. The module also provides CompileResult<T> as
// a convenience type alias for Result<T, CompileError>. These error types enable proper
// error propagation throughout the compiler with descriptive error messages via the
// Display trait implementation provided by thiserror.

//! Error types for the TPDE compiler.
//!
//! Using thiserror for more idiomatic error handling.

use thiserror::Error;

/// Main error type for LLVM compilation.
#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Unsupported {width}-bit {operation} instruction")]
    UnsupportedWidth {
        operation: &'static str,
        width: u32,
    },
    
    #[error("Unsupported instruction: {opcode:?}")]
    UnsupportedInstruction {
        opcode: inkwell::values::InstructionOpcode,
    },
    
    #[error("Register allocation failed: {reason}")]
    RegisterAllocation {
        reason: String,
    },
    
    #[error("Code generation failed: {reason}")]
    CodeGeneration {
        reason: String,
    },
    
    #[error("Function not found: {name}")]
    FunctionNotFound {
        name: String,
    },
    
    #[error("Invalid LLVM value: {reason}")]
    InvalidValue {
        reason: String,
    },
    
    #[error("PHI node cycle detected")]
    PhiCycle,
    
    #[error("Block layout error: {reason}")]
    BlockLayout {
        reason: String,
    },
}

/// Result type alias for compile operations.
pub type CompileResult<T> = Result<T, CompileError>;