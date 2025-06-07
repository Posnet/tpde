// This module provides high-level function code generation that orchestrates x86-64 machine
// code emission with proper calling convention compliance. FunctionCodegen is the main component
// that integrates SysVAssigner (calling convention), X64Encoder (machine code), and FunctionFrame
// (stack layout) to generate complete function bodies. Key responsibilities include processing
// function arguments according to System V ABI (first 6 integers in RDI/RSI/RDX/RCX/R8/R9,
// first 8 floats in XMM0-XMM7, rest on stack), managing return value placement (RAX/RDX for
// integers, XMM0/XMM1 for floats), generating standard prologue (push rbp, mov rbp rsp,
// save callee-saved registers, allocate stack), and matching epilogue. The module supports
// spill slot allocation for register pressure, callee-saved register preservation (RBX, R12-R15),
// and proper stack alignment. ArgInfo provides type descriptions for arguments/returns.
// This abstraction layer ensures all generated functions are ABI-compliant and can correctly
// interoperate with C code, handling both simple cases and complex scenarios with many
// arguments or mixed integer/floating-point parameters.

//! Function code generation with calling conventions.
//!
//! This module provides the bridge between calling conventions and machine code
//! generation, implementing proper function prologue/epilogue generation according
//! to the System V x86-64 ABI.

use crate::{
    x64::{
        calling_convention::{CCAssigner, SysVAssigner, FunctionFrame, CCAssignment, RegBank},
        encoder::{X64Encoder, EncodingError},
    },
    core::register_file::AsmReg,
};

/// Function code generator that handles calling conventions.
///
/// This integrates the calling convention logic with machine code generation,
/// providing a high-level interface for compiling functions with proper
/// ABI compliance.
pub struct FunctionCodegen {
    /// Machine code encoder.
    encoder: X64Encoder,
    /// Calling convention assigner.
    cc_assigner: SysVAssigner,
    /// Stack frame management.
    frame: FunctionFrame,
    /// Whether function makes calls (affects register allocation).
    makes_calls: bool,
}

impl FunctionCodegen {
    /// Create a new function code generator.
    pub fn new() -> Result<Self, FunctionCodegenError> {
        Ok(Self {
            encoder: X64Encoder::new()?,
            cc_assigner: SysVAssigner::new(),
            frame: FunctionFrame::new(),
            makes_calls: false,
        })
    }
    
    /// Mark that this function makes calls (affects callee-saved register handling).
    pub fn set_makes_calls(&mut self, makes_calls: bool) {
        self.makes_calls = makes_calls;
    }
    
    /// Process function arguments according to calling convention.
    pub fn process_arguments(&mut self, args: &[ArgInfo]) -> Result<Vec<CCAssignment>, FunctionCodegenError> {
        self.cc_assigner.reset();
        let mut assignments = Vec::new();
        
        for arg in args {
            let mut assignment = CCAssignment::new(arg.bank, arg.size, arg.align);
            self.cc_assigner.assign_arg(&mut assignment);
            assignments.push(assignment);
        }
        
        self.frame.arg_assignments = assignments.clone();
        Ok(assignments)
    }
    
    /// Process return values according to calling convention.
    pub fn process_return_values(&mut self, rets: &[ArgInfo]) -> Result<Vec<CCAssignment>, FunctionCodegenError> {
        let mut assignments = Vec::new();
        
        for ret in rets {
            let mut assignment = CCAssignment::new(ret.bank, ret.size, ret.align);
            self.cc_assigner.assign_ret(&mut assignment);
            assignments.push(assignment);
        }
        
        self.frame.ret_assignments = assignments.clone();
        Ok(assignments)
    }
    
    /// Add a callee-saved register that needs to be preserved.
    pub fn add_callee_saved_register(&mut self, reg: AsmReg) {
        self.frame.add_saved_register(reg);
    }
    
    /// Generate function prologue.
    ///
    /// Emits the standard System V x86-64 function prologue:
    /// - push rbp
    /// - mov rbp, rsp
    /// - sub rsp, frame_size (if needed)
    /// - save callee-saved registers
    pub fn emit_prologue(&mut self) -> Result<(), FunctionCodegenError> {
        // Calculate frame size first
        self.frame.calculate_frame_size();
        
        // Standard prologue: push rbp; mov rbp, rsp
        let rbp = AsmReg::new(0, 5);
        let rsp = AsmReg::new(0, 4);
        
        self.encoder.push_reg(rbp)?;
        self.encoder.mov_reg_reg(rbp, rsp)?;
        
        // Save callee-saved registers
        for &reg in &self.frame.saved_registers {
            self.encoder.push_reg(reg)?;
        }
        
        // Allocate stack space for locals and spills
        if self.frame.frame_size > 0 {
            self.encoder.sub_reg_imm(rsp, self.frame.frame_size as i32)?;
        }
        
        Ok(())
    }
    
    /// Generate function epilogue.
    ///
    /// Emits the standard System V x86-64 function epilogue:
    /// - add rsp, frame_size (if needed)
    /// - restore callee-saved registers
    /// - pop rbp
    /// - ret
    pub fn emit_epilogue(&mut self) -> Result<(), FunctionCodegenError> {
        let rbp = AsmReg::new(0, 5);
        let rsp = AsmReg::new(0, 4);
        
        // Deallocate stack space
        if self.frame.frame_size > 0 {
            self.encoder.add_reg_imm(rsp, self.frame.frame_size as i32)?;
        }
        
        // Restore callee-saved registers (in reverse order)
        for &reg in self.frame.saved_registers.iter().rev() {
            self.encoder.pop_reg(reg)?;
        }
        
        // Standard epilogue: pop rbp; ret
        self.encoder.pop_reg(rbp)?;
        self.encoder.ret()?;
        
        Ok(())
    }
    
    /// Emit code to load an argument from its assigned location.
    pub fn emit_load_argument(&mut self, arg_index: usize, dest_reg: AsmReg) -> Result<(), FunctionCodegenError> {
        if arg_index >= self.frame.arg_assignments.len() {
            return Err(FunctionCodegenError::InvalidArgument);
        }
        
        let assignment = &self.frame.arg_assignments[arg_index];
        
        if let Some(src_reg) = assignment.reg {
            // Argument is in register - move if different
            if src_reg != dest_reg {
                self.encoder.mov_reg_reg(dest_reg, src_reg)?;
            }
        } else if let Some(stack_off) = assignment.stack_off {
            // Argument is on stack - load from [rbp + 16 + offset]
            // (16 = 8 bytes return address + 8 bytes saved rbp)
            let rbp = AsmReg::new(0, 5);
            let offset = 16 + stack_off;
            self.encoder.mov_reg_mem(dest_reg, rbp, offset)?;
        } else {
            return Err(FunctionCodegenError::InvalidArgument);
        }
        
        Ok(())
    }
    
    /// Emit code to store a return value to its assigned location.
    pub fn emit_store_return_value(&mut self, ret_index: usize, src_reg: AsmReg) -> Result<(), FunctionCodegenError> {
        if ret_index >= self.frame.ret_assignments.len() {
            return Err(FunctionCodegenError::InvalidReturnValue);
        }
        
        let assignment = &self.frame.ret_assignments[ret_index];
        
        if let Some(dest_reg) = assignment.reg {
            // Return value goes in register
            if src_reg != dest_reg {
                self.encoder.mov_reg_reg(dest_reg, src_reg)?;
            }
        } else {
            // Return values should always be in registers for System V ABI
            return Err(FunctionCodegenError::InvalidReturnValue);
        }
        
        Ok(())
    }
    
    /// Allocate a spill slot and return its frame offset.
    pub fn allocate_spill_slot(&mut self, size: u32) -> i32 {
        self.frame.allocate_spill_slot(size)
    }
    
    /// Emit code to spill a register to its assigned stack slot.
    pub fn emit_spill(&mut self, reg: AsmReg, spill_offset: i32) -> Result<(), FunctionCodegenError> {
        let rbp = AsmReg::new(0, 5);
        self.encoder.mov_mem_reg(rbp, spill_offset, reg)?;
        Ok(())
    }
    
    /// Emit code to reload a register from its spill slot.
    pub fn emit_reload(&mut self, reg: AsmReg, spill_offset: i32) -> Result<(), FunctionCodegenError> {
        let rbp = AsmReg::new(0, 5);
        self.encoder.mov_reg_mem(reg, rbp, spill_offset)?;
        Ok(())
    }
    
    /// Get access to the underlying encoder for instruction emission.
    pub fn encoder_mut(&mut self) -> &mut X64Encoder {
        &mut self.encoder
    }
    
    /// Get the calling convention information.
    pub fn get_cc_info(&self) -> &crate::x64::calling_convention::CCInfo {
        self.cc_assigner.get_ccinfo()
    }
    
    /// Get the current stack frame info.
    pub fn get_frame(&self) -> &FunctionFrame {
        &self.frame
    }
    
    /// Finalize code generation and return machine code.
    pub fn finalize(mut self) -> Result<Vec<u8>, FunctionCodegenError> {
        Ok(self.encoder.finalize()?)
    }
}

/// Information about a function argument or return value.
#[derive(Debug, Clone)]
pub struct ArgInfo {
    /// Register bank (GP or XMM).
    pub bank: RegBank,
    /// Size in bytes.
    pub size: u32,
    /// Alignment requirement.
    pub align: u32,
}

impl ArgInfo {
    /// Create argument info for a 64-bit integer.
    pub fn int64() -> Self {
        Self {
            bank: RegBank::GeneralPurpose,
            size: 8,
            align: 8,
        }
    }
    
    /// Create argument info for a 32-bit integer.
    pub fn int32() -> Self {
        Self {
            bank: RegBank::GeneralPurpose,
            size: 4,
            align: 4,
        }
    }
    
    /// Create argument info for a 16-bit integer.
    pub fn int16() -> Self {
        Self {
            bank: RegBank::GeneralPurpose,
            size: 2,
            align: 2,
        }
    }
    
    /// Create argument info for an 8-bit integer.
    pub fn int8() -> Self {
        Self {
            bank: RegBank::GeneralPurpose,
            size: 1,
            align: 1,
        }
    }
    
    /// Create argument info for a 64-bit float.
    pub fn float64() -> Self {
        Self {
            bank: RegBank::Xmm,
            size: 8,
            align: 8,
        }
    }
    
    /// Create argument info for a 32-bit float.
    pub fn float32() -> Self {
        Self {
            bank: RegBank::Xmm,
            size: 4,
            align: 4,
        }
    }
}

/// Errors that can occur during function code generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionCodegenError {
    /// Encoding error from the x64 encoder.
    Encoding(EncodingError),
    /// Invalid argument index.
    InvalidArgument,
    /// Invalid return value configuration.
    InvalidReturnValue,
    /// Stack frame layout error.
    FrameLayoutError,
}

impl From<EncodingError> for FunctionCodegenError {
    fn from(err: EncodingError) -> Self {
        Self::Encoding(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function_codegen() {
        let mut codegen = FunctionCodegen::new().unwrap();
        
        // Simple function: int add(int a, int b) { return a + b; }
        let args = vec![ArgInfo::int32(), ArgInfo::int32()];
        let rets = vec![ArgInfo::int32()];
        
        let arg_assignments = codegen.process_arguments(&args).unwrap();
        let ret_assignments = codegen.process_return_values(&rets).unwrap();
        
        // Verify arguments are assigned to RDI, RSI
        assert!(arg_assignments[0].reg.is_some());
        assert!(arg_assignments[1].reg.is_some());
        assert_eq!(arg_assignments[0].reg.unwrap(), AsmReg::new(0, 7)); // RDI
        assert_eq!(arg_assignments[1].reg.unwrap(), AsmReg::new(0, 6)); // RSI
        
        // Verify return value is assigned to RAX
        assert!(ret_assignments[0].reg.is_some());
        assert_eq!(ret_assignments[0].reg.unwrap(), AsmReg::new(0, 0)); // RAX
        
        // Generate prologue
        codegen.emit_prologue().unwrap();
        
        // Generate epilogue
        codegen.emit_epilogue().unwrap();
        
        // Should produce valid machine code
        let code = codegen.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_callee_saved_registers() {
        let mut codegen = FunctionCodegen::new().unwrap();
        codegen.set_makes_calls(true);
        
        // Add some callee-saved registers
        let rbx = AsmReg::new(0, 3);
        let r12 = AsmReg::new(0, 12);
        
        codegen.add_callee_saved_register(rbx);
        codegen.add_callee_saved_register(r12);
        
        // Generate prologue (should save registers)
        codegen.emit_prologue().unwrap();
        
        // Generate epilogue (should restore registers)
        codegen.emit_epilogue().unwrap();
        
        // Check frame before finalization
        let frame = codegen.get_frame();
        assert_eq!(frame.saved_registers.len(), 2);
        
        let code = codegen.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_spill_slot_allocation() {
        let mut codegen = FunctionCodegen::new().unwrap();
        
        // Allocate some spill slots
        let slot1 = codegen.allocate_spill_slot(8);
        let slot2 = codegen.allocate_spill_slot(4);
        
        // Slots should be different
        assert_ne!(slot1, slot2);
        
        // Should be able to emit spill/reload code
        let reg = AsmReg::new(0, 0); // RAX
        codegen.emit_spill(reg, slot1).unwrap();
        codegen.emit_reload(reg, slot1).unwrap();
    }

    #[test]
    fn test_many_arguments() {
        let mut codegen = FunctionCodegen::new().unwrap();
        
        // Function with 8 integer arguments (6 in regs, 2 on stack)
        let args = vec![
            ArgInfo::int64(), ArgInfo::int64(), ArgInfo::int64(),
            ArgInfo::int64(), ArgInfo::int64(), ArgInfo::int64(),
            ArgInfo::int64(), ArgInfo::int64(),
        ];
        
        let assignments = codegen.process_arguments(&args).unwrap();
        
        // First 6 should be in registers
        for assignment in assignments.iter().take(6) {
            assert!(assignment.reg.is_some());
            assert!(assignment.stack_off.is_none());
        }
        
        // Last 2 should be on stack
        for assignment in assignments.iter().skip(6).take(2) {
            assert!(assignment.reg.is_none());
            assert!(assignment.stack_off.is_some());
        }
    }

    #[test]
    fn test_mixed_argument_types() {
        let mut codegen = FunctionCodegen::new().unwrap();
        
        // Function with mixed int/float arguments
        let args = vec![
            ArgInfo::int64(),    // RDI
            ArgInfo::float64(),  // XMM0
            ArgInfo::int32(),    // RSI
            ArgInfo::float32(),  // XMM1
        ];
        
        let assignments = codegen.process_arguments(&args).unwrap();
        
        // Verify register assignments
        assert_eq!(assignments[0].reg.unwrap(), AsmReg::new(0, 7)); // RDI
        assert_eq!(assignments[1].reg.unwrap(), AsmReg::new(1, 0)); // XMM0
        assert_eq!(assignments[2].reg.unwrap(), AsmReg::new(0, 6)); // RSI
        assert_eq!(assignments[3].reg.unwrap(), AsmReg::new(1, 1)); // XMM1
    }
}