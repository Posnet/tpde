//! LLVM-specific compiler implementation with proper instruction selection.
//!
//! This module provides a specialized compiler for LLVM IR that uses opcode-based
//! instruction selection instead of the generic operand-counting approach. 

/// Categories of LLVM instructions for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionCategory {
    /// Arithmetic operations (add, sub, mul, div, etc.).
    Arithmetic,
    /// Comparison operations (icmp, fcmp).
    Comparison,
    /// Memory operations (load, store, alloca).
    Memory,
    /// Control flow (br, switch, call, ret).
    ControlFlow,
    /// PHI nodes for SSA form.
    Phi,
    /// Type conversion operations.
    Conversion,
    /// Other/unsupported operations.
    Other,
}

/// Interface for LLVM adaptors that provide instruction opcode information.
///
/// This trait extends the basic IrAdaptor functionality with LLVM-specific
/// methods needed for proper instruction selection.
pub trait LlvmAdaptorInterface: crate::adaptor::IrAdaptor {
    /// Get instruction opcode information for classification.
    fn get_instruction_category(&self, inst: Self::InstRef) -> InstructionCategory;
    
    /// Get comparison predicate for ICMP instructions.
    fn get_icmp_predicate(&self, inst: Self::InstRef) -> Option<String>;
    
    /// Check if the current function makes calls.
    fn makes_calls(&self) -> bool;
    
    /// Check if the current function has allocas.
    fn has_allocas(&self) -> bool;
}

