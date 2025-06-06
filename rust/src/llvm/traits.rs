// This module defines LLVM-specific traits and types that extend the generic IrAdaptor
// interface with LLVM-specific functionality. InstructionCategory provides a classification
// system for LLVM instructions (Arithmetic, Comparison, Memory, ControlFlow, Phi, Conversion,
// Other) that enables proper instruction selection in the compiler. LlvmAdaptorInterface
// trait extends IrAdaptor with methods to extract LLVM-specific information: instruction
// categorization by opcode, ICMP predicate extraction for comparison instructions, and
// function property queries (makes_calls, has_allocas). This trait was part of the
// enhanced adaptor integration but revealed trait bound complexity issues during the
// architectural redesign, leading to the shift to concrete LLVM types. The types remain
// useful for instruction dispatch even in the concrete implementation.

//! LLVM-specific traits and types.
//!
//! This module defines the interfaces and types specific to LLVM IR compilation.

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
pub trait LlvmAdaptorInterface: crate::core::IrAdaptor {
    /// Get instruction opcode information for classification.
    fn get_instruction_category(&self, inst: <Self as crate::core::IrAdaptor>::InstRef) -> InstructionCategory;
    
    /// Get comparison predicate for ICMP instructions.
    fn get_icmp_predicate(&self, inst: <Self as crate::core::IrAdaptor>::InstRef) -> Option<String>;
    
    /// Check if the current function makes calls.
    fn makes_calls(&self) -> bool;
    
    /// Check if the current function has allocas.
    fn has_allocas(&self) -> bool;
}