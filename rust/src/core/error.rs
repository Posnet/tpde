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