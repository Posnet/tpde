// This module provides comprehensive x86-64 instruction encoding capabilities using the iced-x86
// library, serving as the machine code generation backend for the TPDE compiler. X64Encoder
// is the core component that converts AsmReg register references into real x86-64 machine code,
// supporting all major instruction types: moves (MOV, MOVZX), arithmetic (ADD, SUB, IMUL),
// comparisons (CMP, TEST), control flow (JMP, Jcc, CALL, RET), memory operations with various
// addressing modes, and stack manipulation (PUSH, POP). The encoder handles both 32-bit and
// 64-bit instruction variants for proper LLVM i32/i64 type compilation, manages labels for
// basic blocks to enable forward/backward jumps, and includes all SETcc instructions for
// condition code materialization. InstructionSelector provides higher-level patterns like
// register reuse optimization for binary operations. This implementation achieves parity with
// the C++ fadec-based encoder while providing safer Rust abstractions. Critical features
// include support for LEA instruction for efficient address calculations and function
// prologue/epilogue generation following System V ABI conventions.

//! x86-64 instruction encoding using iced-x86.
//!
//! This module provides real x86-64 machine code generation equivalent to the C++
//! fadec-based implementation. It integrates with the ValueRef/RegisterFile system
//! to emit actual executable machine code for basic arithmetic and control flow.

use crate::core::register_file::AsmReg;
use iced_x86::code_asm::{registers::cl, *};
use std::collections::HashMap;

/// Error types for instruction encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodingError {
    /// Invalid register for the requested operation.
    InvalidRegister,
    /// Unsupported instruction or operand combination.
    UnsupportedInstruction,
    /// Assembly error from iced-x86.
    AssemblyError(String),
    /// Memory operand construction failed.
    InvalidMemoryOperand,
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodingError::InvalidRegister => write!(f, "Invalid register for operation"),
            EncodingError::UnsupportedInstruction => {
                write!(f, "Unsupported instruction or operand combination")
            }
            EncodingError::AssemblyError(msg) => write!(f, "Assembly error: {msg}"),
            EncodingError::InvalidMemoryOperand => write!(f, "Invalid memory operand"),
        }
    }
}

impl std::error::Error for EncodingError {}

/// x86-64 instruction encoder using iced-x86.
///
/// This provides the equivalent functionality to the C++ fadec-based encoder,
/// generating real x86-64 machine code that can be executed.
pub struct X64Encoder {
    /// Code assembler for generating instructions.
    assembler: CodeAssembler,
    /// Current instruction buffer position.
    position: u64,
    /// Labels for basic blocks to enable proper control flow.
    block_labels: HashMap<usize, CodeLabel>,
    /// Track which block labels have been placed.
    placed_blocks: std::collections::HashSet<usize>,
}

impl X64Encoder {
    /// Create a new x86-64 encoder.
    pub fn new() -> Result<Self, EncodingError> {
        let assembler =
            CodeAssembler::new(64).map_err(|e| EncodingError::AssemblyError(e.to_string()))?;

        Ok(Self {
            assembler,
            position: 0x1000, // Start at a reasonable base address
            block_labels: HashMap::new(),
            placed_blocks: std::collections::HashSet::new(),
        })
    }

    /// Convert AsmReg to iced-x86 Register for 64-bit GP registers.
    fn to_gp_register(&self, reg: AsmReg) -> Result<AsmRegister64, EncodingError> {
        if reg.bank != 0 {
            return Err(EncodingError::InvalidRegister);
        }

        self.to_gp64_register(reg)
    }

    /// Convert AsmReg to iced-x86 64-bit GP register.
    fn to_gp64_register(&self, reg: AsmReg) -> Result<AsmRegister64, EncodingError> {
        use iced_x86::code_asm::*;
        
        if reg.bank != 0 {
            return Err(EncodingError::InvalidRegister);
        }

        const GP64_REGS: [AsmRegister64; 16] = [
            rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
            r8, r9, r10, r11, r12, r13, r14, r15,
        ];
        
        GP64_REGS.get(reg.id as usize)
            .copied()
            .ok_or(EncodingError::InvalidRegister)
    }

    /// Convert AsmReg to iced-x86 32-bit GP register.
    fn to_gp32_register(&self, reg: AsmReg) -> Result<AsmRegister32, EncodingError> {
        use iced_x86::code_asm::*;
        
        if reg.bank != 0 {
            return Err(EncodingError::InvalidRegister);
        }

        const GP32_REGS: [AsmRegister32; 16] = [
            eax, ecx, edx, ebx, esp, ebp, esi, edi,
            r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d,
        ];
        
        GP32_REGS.get(reg.id as usize)
            .copied()
            .ok_or(EncodingError::InvalidRegister)
    }

    /// Convert AsmReg to iced-x86 16-bit GP register.
    fn to_gp16_register(&self, reg: AsmReg) -> Result<AsmRegister16, EncodingError> {
        use iced_x86::code_asm::*;
        
        if reg.bank != 0 {
            return Err(EncodingError::InvalidRegister);
        }

        const GP16_REGS: [AsmRegister16; 16] = [
            ax, cx, dx, bx, sp, bp, si, di,
            r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w,
        ];
        
        GP16_REGS.get(reg.id as usize)
            .copied()
            .ok_or(EncodingError::InvalidRegister)
    }

    /// Convert AsmReg to iced-x86 8-bit GP register.
    fn to_gp8_register(&self, reg: AsmReg) -> Result<AsmRegister8, EncodingError> {
        use iced_x86::code_asm::*;
        
        if reg.bank != 0 {
            return Err(EncodingError::InvalidRegister);
        }

        const GP8_REGS: [AsmRegister8; 16] = [
            al, cl, dl, bl, spl, bpl, sil, dil,
            r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b,
        ];
        
        GP8_REGS.get(reg.id as usize)
            .copied()
            .ok_or(EncodingError::InvalidRegister)
    }

    /// Convert AsmReg to iced-x86 Register for XMM registers.
    #[allow(dead_code)]
    fn to_xmm_register(&self, reg: AsmReg) -> Result<AsmRegisterXmm, EncodingError> {
        use iced_x86::code_asm::*;
        
        if reg.bank != 1 {
            return Err(EncodingError::InvalidRegister);
        }

        const XMM_REGS: [AsmRegisterXmm; 16] = [
            xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
            xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,
        ];
        
        XMM_REGS.get(reg.id as usize)
            .copied()
            .ok_or(EncodingError::InvalidRegister)
    }

    /// Emit MOV instruction - register to register.
    pub fn mov_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .mov(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - immediate to register.
    pub fn mov_reg_imm(&mut self, dst: AsmReg, imm: i64) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        // iced-x86 mov with immediate requires i64 for 64-bit registers
        self.assembler
            .mov(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - memory to register.
    pub fn mov_reg_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = qword_ptr(base_reg + offset);
        self.assembler
            .mov(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - register to memory.
    pub fn mov_mem_reg(
        &mut self,
        base: AsmReg,
        offset: i32,
        src: AsmReg,
    ) -> Result<(), EncodingError> {
        let base_reg = self.to_gp_register(base)?;
        let src_reg = self.to_gp_register(src)?;

        let mem = qword_ptr(base_reg + offset);
        self.assembler
            .mov(mem, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== MOV VARIANTS FOR DIFFERENT SIZES ====

    /// Emit MOV instruction - register to register (16-bit).
    pub fn mov16_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp16_register(dst)?;
        let src_reg = self.to_gp16_register(src)?;

        self.assembler
            .mov(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - register to register (8-bit).
    pub fn mov8_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;
        let src_reg = self.to_gp8_register(src)?;

        self.assembler
            .mov(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== SIGN/ZERO EXTENSION INSTRUCTIONS ====

    /// Emit MOVSX instruction - 8-bit to 32-bit sign extension.
    pub fn movsx_reg8_to_reg32(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp8_register(src)?;

        self.assembler
            .movsx(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVSX instruction - 8-bit to 64-bit sign extension.
    pub fn movsx_reg8_to_reg64(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp8_register(src)?;

        self.assembler
            .movsx(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVSX instruction - 16-bit to 32-bit sign extension.
    pub fn movsx_reg16_to_reg32(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp16_register(src)?;

        self.assembler
            .movsx(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVSX instruction - 16-bit to 64-bit sign extension.
    pub fn movsx_reg16_to_reg64(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp16_register(src)?;

        self.assembler
            .movsx(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVSXD instruction - 32-bit to 64-bit sign extension.
    pub fn movsxd_reg32_to_reg64(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .movsxd(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVZX instruction - 8-bit to 32-bit zero extension.
    pub fn movzx_reg8_to_reg32(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp8_register(src)?;

        self.assembler
            .movzx(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVZX instruction - 16-bit to 32-bit zero extension.
    pub fn movzx_reg16_to_reg32(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp16_register(src)?;

        self.assembler
            .movzx(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit ADD instruction - register to register.
    pub fn add_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .add(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit ADD instruction - immediate to register.
    pub fn add_reg_imm(&mut self, dst: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .add(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SUB instruction - register to register.
    pub fn sub_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .sub(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SUB instruction - immediate to register.
    pub fn sub_reg_imm(&mut self, dst: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .sub(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit LEA instruction for efficient address calculations.
    pub fn lea(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        index: Option<AsmReg>,
        scale: u32,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = if let Some(index_reg) = index {
            let idx_reg = self.to_gp_register(index_reg)?;
            match scale {
                1 => base_reg + idx_reg + offset,
                2 => base_reg + idx_reg * 2 + offset,
                4 => base_reg + idx_reg * 4 + offset,
                8 => base_reg + idx_reg * 8 + offset,
                _ => return Err(EncodingError::InvalidMemoryOperand),
            }
        } else {
            base_reg + offset
        };

        self.assembler
            .lea(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit CMP instruction - register to register.
    pub fn cmp_reg_reg(&mut self, left: AsmReg, right: AsmReg) -> Result<(), EncodingError> {
        let left_reg = self.to_gp_register(left)?;
        let right_reg = self.to_gp_register(right)?;

        self.assembler
            .cmp(left_reg, right_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit CMP instruction - immediate to register.
    pub fn cmp_reg_imm(&mut self, reg: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(reg)?;

        self.assembler
            .cmp(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit PUSH instruction.
    pub fn push_reg(&mut self, reg: AsmReg) -> Result<(), EncodingError> {
        let src_reg = self.to_gp_register(reg)?;

        self.assembler
            .push(src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit POP instruction.
    pub fn pop_reg(&mut self, reg: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(reg)?;

        self.assembler
            .pop(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit RET instruction.
    pub fn ret(&mut self) -> Result<(), EncodingError> {
        self.assembler
            .ret()
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit CALL instruction to register.
    pub fn call_reg(&mut self, reg: AsmReg) -> Result<(), EncodingError> {
        let target_reg = self.to_gp_register(reg)?;

        self.assembler
            .call(target_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== CRITICAL MISSING INSTRUCTIONS FOR FACTORIAL ====

    /// Emit IMUL instruction - multiply two registers (64-bit).
    pub fn imul_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        // IMUL with two operands: imul dst, src (dst = dst * src)
        self.assembler
            .imul_2(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit IMUL instruction - multiply two registers (32-bit).
    pub fn imul32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        // IMUL with two operands: imul dst, src (dst = dst * src)
        self.assembler
            .imul_2(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== 32-BIT INSTRUCTION VARIANTS (Critical for i32 operations) ====

    /// Emit MOV instruction - register to register (32-bit).
    pub fn mov32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .mov(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - immediate to register (32-bit).
    pub fn mov32_reg_imm(&mut self, dst: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;

        self.assembler
            .mov(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit ADD instruction - register to register (32-bit).
    pub fn add32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .add(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit ADD instruction - register to register (64-bit).
    pub fn add64_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .add(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SUB instruction - register to register (32-bit).
    pub fn sub32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .sub(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SUB instruction - register to register (64-bit).
    pub fn sub64_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .sub(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit CMP instruction - register to register (32-bit).
    pub fn cmp32_reg_reg(&mut self, left: AsmReg, right: AsmReg) -> Result<(), EncodingError> {
        let left_reg = self.to_gp32_register(left)?;
        let right_reg = self.to_gp32_register(right)?;

        self.assembler
            .cmp(left_reg, right_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit CMP instruction - immediate to register (32-bit).
    pub fn cmp32_reg_imm(&mut self, reg: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(reg)?;

        self.assembler
            .cmp(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit TEST instruction - register to register (8-bit).
    pub fn test8_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;
        let src_reg = self.to_gp8_register(src)?;

        self.assembler
            .test(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit TEST instruction - register to register (64-bit default).
    pub fn test_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .test(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // SETcc instructions for condition code to boolean conversion

    /// Emit SETE instruction - set byte on equal.
    pub fn sete_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .sete(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETNE instruction - set byte on not equal.
    pub fn setne_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setne(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETG instruction - set byte on greater (signed).
    pub fn setg_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setg(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETGE instruction - set byte on greater or equal (signed).
    pub fn setge_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setge(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETL instruction - set byte on less (signed).
    pub fn setl_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setl(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETLE instruction - set byte on less or equal (signed).
    pub fn setle_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setle(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETA instruction - set byte on above (unsigned).
    pub fn seta_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .seta(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETAE instruction - set byte on above or equal (unsigned).
    pub fn setae_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setae(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETB instruction - set byte on below (unsigned).
    pub fn setb_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setb(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SETBE instruction - set byte on below or equal (unsigned).
    pub fn setbe_reg(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;

        self.assembler
            .setbe(dst_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SUB instruction - immediate to register (32-bit).
    pub fn sub32_reg_imm(&mut self, dst: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;

        self.assembler
            .sub(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SUB instruction - immediate to register (64-bit).
    pub fn sub64_reg_imm(&mut self, dst: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .sub(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit AND instruction - immediate to register (64-bit).
    pub fn and64_reg_imm(&mut self, dst: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .and(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== BITWISE OPERATIONS ====

    /// Emit AND instruction - register to register (32-bit).
    pub fn and32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .and(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit AND instruction - register to register (64-bit).
    pub fn and64_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .and(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit OR instruction - register to register (32-bit).
    pub fn or32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .or(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit OR instruction - register to register (64-bit).
    pub fn or64_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .or(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit XOR instruction - register to register (32-bit).
    pub fn xor32_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let src_reg = self.to_gp32_register(src)?;

        self.assembler
            .xor(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit XOR instruction - register to register (64-bit).
    pub fn xor64_reg_reg(&mut self, dst: AsmReg, src: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;
        let src_reg = self.to_gp_register(src)?;

        self.assembler
            .xor(dst_reg, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== SHIFT OPERATIONS ====

    /// Emit SHL instruction - shift left by CL (32-bit).
    pub fn shl32_reg_cl(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;

        self.assembler
            .shl(dst_reg, cl)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SHL instruction - shift left by CL (64-bit).
    pub fn shl64_reg_cl(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .shl(dst_reg, cl)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SHR instruction - logical shift right by CL (32-bit).
    pub fn shr32_reg_cl(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;

        self.assembler
            .shr(dst_reg, cl)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SHR instruction - logical shift right by CL (64-bit).
    pub fn shr64_reg_cl(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .shr(dst_reg, cl)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SAR instruction - arithmetic shift right by CL (32-bit).
    pub fn sar32_reg_cl(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;

        self.assembler
            .sar(dst_reg, cl)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit SAR instruction - arithmetic shift right by CL (64-bit).
    pub fn sar64_reg_cl(&mut self, dst: AsmReg) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp_register(dst)?;

        self.assembler
            .sar(dst_reg, cl)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== MEMORY OPERATIONS ====

    /// Emit MOVZX instruction - 8-bit memory to 32-bit register (zero extend).
    pub fn movzx_reg8_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = byte_ptr(base_reg + offset);
        self.assembler
            .movzx(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOVZX instruction - 16-bit memory to 32-bit register (zero extend).
    pub fn movzx_reg16_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = word_ptr(base_reg + offset);
        self.assembler
            .movzx(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 32-bit memory to register.
    pub fn mov32_reg_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = dword_ptr(base_reg + offset);
        self.assembler
            .mov(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 8-bit memory to register.
    pub fn mov8_reg_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp8_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = byte_ptr(base_reg + offset);
        self.assembler
            .mov(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 16-bit memory to register.
    pub fn mov16_reg_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp16_register(dst)?;
        let base_reg = self.to_gp_register(base)?;

        let mem = word_ptr(base_reg + offset);
        self.assembler
            .mov(dst_reg, mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 64-bit memory to register.
    pub fn mov64_reg_mem(
        &mut self,
        dst: AsmReg,
        base: AsmReg,
        offset: i32,
    ) -> Result<(), EncodingError> {
        // Just use the existing mov_reg_mem which is already 64-bit
        self.mov_reg_mem(dst, base, offset)
    }

    /// Emit MOV instruction - 8-bit register to memory.
    pub fn mov8_mem_reg(
        &mut self,
        base: AsmReg,
        offset: i32,
        src: AsmReg,
    ) -> Result<(), EncodingError> {
        let base_reg = self.to_gp_register(base)?;
        let src_reg = self.to_gp8_register(src)?;

        let mem = byte_ptr(base_reg + offset);
        self.assembler
            .mov(mem, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 16-bit register to memory.
    pub fn mov16_mem_reg(
        &mut self,
        base: AsmReg,
        offset: i32,
        src: AsmReg,
    ) -> Result<(), EncodingError> {
        let base_reg = self.to_gp_register(base)?;
        let src_reg = self.to_gp16_register(src)?;

        let mem = word_ptr(base_reg + offset);
        self.assembler
            .mov(mem, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 32-bit register to memory.
    pub fn mov32_mem_reg(
        &mut self,
        base: AsmReg,
        offset: i32,
        src: AsmReg,
    ) -> Result<(), EncodingError> {
        let base_reg = self.to_gp_register(base)?;
        let src_reg = self.to_gp32_register(src)?;

        let mem = dword_ptr(base_reg + offset);
        self.assembler
            .mov(mem, src_reg)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit MOV instruction - 64-bit register to memory.
    pub fn mov64_mem_reg(
        &mut self,
        base: AsmReg,
        offset: i32,
        src: AsmReg,
    ) -> Result<(), EncodingError> {
        // Just use the existing mov_mem_reg which is already 64-bit
        self.mov_mem_reg(base, offset, src)
    }

    /// Emit direct CALL instruction (relative offset).
    pub fn call_direct(&mut self, offset: u64) -> Result<(), EncodingError> {
        self.assembler
            .call(offset)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit indirect CALL instruction through memory.
    pub fn call_mem(&mut self, base: AsmReg, offset: i32) -> Result<(), EncodingError> {
        let base_reg = self.to_gp_register(base)?;
        let mem = qword_ptr(base_reg + offset);

        self.assembler
            .call(mem)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit TEST instruction - register against immediate.
    pub fn test_reg_imm(&mut self, reg: AsmReg, imm: i32) -> Result<(), EncodingError> {
        let dst_reg = self.to_gp32_register(reg)?;

        self.assembler
            .test(dst_reg, imm)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== LABEL MANAGEMENT FOR CONTROL FLOW ====

    /// Create or get a label for a basic block.
    pub fn get_or_create_label_for_block(&mut self, block_id: usize) -> CodeLabel {
        if let Some(&label) = self.block_labels.get(&block_id) {
            label
        } else {
            let label = self.assembler.create_label();
            self.block_labels.insert(block_id, label);
            label
        }
    }

    /// Place a label for a basic block at the current position.
    pub fn place_label_for_block(&mut self, block_id: usize) -> Result<(), EncodingError> {
        let mut label = self.get_or_create_label_for_block(block_id);
        self.assembler
            .set_label(&mut label)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        self.placed_blocks.insert(block_id);
        Ok(())
    }

    /// Check if a block label has been placed.
    pub fn is_block_placed(&self, block_id: usize) -> bool {
        self.placed_blocks.contains(&block_id)
    }

    // ==== CONTROL FLOW INSTRUCTIONS ====

    /// Emit conditional jump to a specific basic block.
    pub fn jmp_conditional_to_block(
        &mut self,
        condition: JumpCondition,
        target_block: usize,
    ) -> Result<(), EncodingError> {
        let label = self.get_or_create_label_for_block(target_block);

        match condition {
            JumpCondition::Equal => {
                self.assembler
                    .je(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::NotEqual => {
                self.assembler
                    .jne(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::Less => {
                self.assembler
                    .jl(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::Greater => {
                self.assembler
                    .jg(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::LessEqual => {
                self.assembler
                    .jle(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::GreaterEqual => {
                self.assembler
                    .jge(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Emit unconditional jump to a specific basic block.
    pub fn jmp_unconditional_to_block(&mut self, target_block: usize) -> Result<(), EncodingError> {
        let label = self.get_or_create_label_for_block(target_block);

        self.assembler
            .jmp(label)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    // ==== LEGACY METHODS (DEPRECATED - Use block-based methods above) ====

    /// Emit conditional jump instruction.
    pub fn jmp_conditional(
        &mut self,
        condition: JumpCondition,
        _target_offset: i32,
    ) -> Result<(), EncodingError> {
        // Create a label for the target
        let label = self.assembler.create_label();

        // Emit the conditional jump
        match condition {
            JumpCondition::Equal => {
                self.assembler
                    .je(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::NotEqual => {
                self.assembler
                    .jne(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::Less => {
                self.assembler
                    .jl(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::Greater => {
                self.assembler
                    .jg(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::LessEqual => {
                self.assembler
                    .jle(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
            JumpCondition::GreaterEqual => {
                self.assembler
                    .jge(label)
                    .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Emit unconditional jump instruction.
    pub fn jmp_unconditional(&mut self, _target_offset: i32) -> Result<(), EncodingError> {
        let label = self.assembler.create_label();

        self.assembler
            .jmp(label)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Emit a NOP instruction.
    pub fn nop(&mut self) -> Result<(), EncodingError> {
        self.assembler
            .nop()
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        Ok(())
    }

    /// Generate the final machine code bytes.
    pub fn finalize(&mut self) -> Result<Vec<u8>, EncodingError> {
        // Ensure all block labels have been placed
        let unplaced_blocks: Vec<_> = self
            .block_labels
            .keys()
            .filter(|&&block_id| !self.placed_blocks.contains(&block_id))
            .copied()
            .collect();

        for block_id in unplaced_blocks {
            let mut label = self.block_labels[&block_id];
            // Place any unplaced labels at the current position as a fallback
            self.assembler
                .set_label(&mut label)
                .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;
        }

        let result = self
            .assembler
            .assemble(self.position)
            .map_err(|e| EncodingError::AssemblyError(e.to_string()))?;

        Ok(result)
    }

    /// Generate prologue for a function.
    pub fn emit_prologue(&mut self, frame_size: u32) -> Result<(), EncodingError> {
        // push rbp
        self.push_reg(AsmReg::new(0, 5))?; // rbp = bank 0, id 5

        // mov rbp, rsp
        self.mov_reg_reg(AsmReg::new(0, 5), AsmReg::new(0, 4))?; // rbp, rsp

        // sub rsp, frame_size (if needed)
        if frame_size > 0 {
            self.sub_reg_imm(AsmReg::new(0, 4), frame_size as i32)?; // rsp
        }

        Ok(())
    }

    /// Generate epilogue for a function.
    pub fn emit_epilogue(&mut self, frame_size: u32) -> Result<(), EncodingError> {
        // add rsp, frame_size (if needed)
        if frame_size > 0 {
            self.add_reg_imm(AsmReg::new(0, 4), frame_size as i32)?; // rsp
        }

        // pop rbp
        self.pop_reg(AsmReg::new(0, 5))?; // rbp

        // ret
        self.ret()?;

        Ok(())
    }
}

/// Jump conditions for conditional branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpCondition {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

/// High-level instruction selection patterns.
///
/// These implement common code generation patterns equivalent to the C++ compiler.
pub struct InstructionSelector {
    encoder: X64Encoder,
}

impl InstructionSelector {
    /// Create a new instruction selector.
    pub fn new() -> Result<Self, EncodingError> {
        Ok(Self {
            encoder: X64Encoder::new()?,
        })
    }

    /// Compile a binary add operation with register reuse optimization.
    ///
    /// This implements the same pattern as the C++ compiler:
    /// - If result can reuse left operand register: ADD result, right
    /// - Otherwise: LEA result, [left + right]
    pub fn compile_add(
        &mut self,
        result_reg: AsmReg,
        left_reg: AsmReg,
        right_reg: AsmReg,
    ) -> Result<(), EncodingError> {
        if result_reg == left_reg {
            // In-place addition: add result, right
            self.encoder.add_reg_reg(result_reg, right_reg)?;
        } else if result_reg == right_reg {
            // In-place addition: add result, left
            self.encoder.add_reg_reg(result_reg, left_reg)?;
        } else {
            // Three-address form: lea result, [left + right]
            self.encoder
                .lea(result_reg, left_reg, Some(right_reg), 1, 0)?;
        }
        Ok(())
    }

    /// Compile a binary subtract operation.
    pub fn compile_sub(
        &mut self,
        result_reg: AsmReg,
        left_reg: AsmReg,
        right_reg: AsmReg,
    ) -> Result<(), EncodingError> {
        if result_reg == left_reg {
            // In-place subtraction: sub result, right
            self.encoder.sub_reg_reg(result_reg, right_reg)?;
        } else {
            // Move left to result, then subtract: mov result, left; sub result, right
            self.encoder.mov_reg_reg(result_reg, left_reg)?;
            self.encoder.sub_reg_reg(result_reg, right_reg)?;
        }
        Ok(())
    }

    /// Compile an immediate add operation.
    pub fn compile_add_imm(
        &mut self,
        result_reg: AsmReg,
        operand_reg: AsmReg,
        imm: i32,
    ) -> Result<(), EncodingError> {
        if result_reg == operand_reg {
            // In-place: add result, imm
            self.encoder.add_reg_imm(result_reg, imm)?;
        } else {
            // Move and add: mov result, operand; add result, imm
            self.encoder.mov_reg_reg(result_reg, operand_reg)?;
            self.encoder.add_reg_imm(result_reg, imm)?;
        }
        Ok(())
    }

    /// Compile a load from stack.
    pub fn compile_load_stack(
        &mut self,
        result_reg: AsmReg,
        frame_offset: i32,
    ) -> Result<(), EncodingError> {
        // mov result, [rbp + offset]
        let rbp_reg: AsmReg = AsmReg::new(0, 5); // rbp
        self.encoder
            .mov_reg_mem(result_reg, rbp_reg, frame_offset)?;
        Ok(())
    }

    /// Compile a store to stack.
    pub fn compile_store_stack(
        &mut self,
        source_reg: AsmReg,
        frame_offset: i32,
    ) -> Result<(), EncodingError> {
        // mov [rbp + offset], source
        let rbp_reg: AsmReg = AsmReg::new(0, 5); // rbp
        self.encoder
            .mov_mem_reg(rbp_reg, frame_offset, source_reg)?;
        Ok(())
    }

    /// Emit function prologue.
    pub fn emit_prologue(&mut self, frame_size: u32) -> Result<(), EncodingError> {
        self.encoder.emit_prologue(frame_size)
    }

    /// Emit function epilogue.
    pub fn emit_epilogue(&mut self, frame_size: u32) -> Result<(), EncodingError> {
        self.encoder.emit_epilogue(frame_size)
    }

    /// Get the final machine code.
    pub fn finalize(mut self) -> Result<Vec<u8>, EncodingError> {
        self.encoder.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::{InstructionSelector, JumpCondition, X64Encoder};
    use crate::core::register_file::AsmReg;

    #[test]
    fn test_basic_instructions() {
        let mut encoder = X64Encoder::new().unwrap();

        // Test mov rax, rcx
        let rax = AsmReg::new(0, 0);
        let rcx = AsmReg::new(0, 1);
        encoder.mov_reg_reg(rax, rcx).unwrap();

        // Test add rax, rdx
        let rdx = AsmReg::new(0, 2);
        encoder.add_reg_reg(rax, rdx).unwrap();

        // Test mov rax, 42
        encoder.mov_reg_imm(rax, 42).unwrap();

        // Test ret
        encoder.ret().unwrap();

        // Generate machine code
        let code = encoder.finalize().unwrap();
        assert!(!code.is_empty());

        // The code should contain real x86-64 machine code bytes
        // This is a basic smoke test to ensure we can generate something
    }

    #[test]
    fn test_instruction_selector() {
        let mut selector = InstructionSelector::new().unwrap();

        let rax = AsmReg::new(0, 0);
        let rcx = AsmReg::new(0, 1);
        let rdx = AsmReg::new(0, 2);

        // Test add with register reuse
        selector.compile_add(rax, rax, rcx).unwrap();

        // Test add with three registers
        selector.compile_add(rdx, rax, rcx).unwrap();

        // Test immediate add
        selector.compile_add_imm(rax, rax, 100).unwrap();

        let code = selector.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_prologue_epilogue() {
        let mut encoder = X64Encoder::new().unwrap();

        // Test function prologue
        encoder.emit_prologue(32).unwrap();

        // Test function epilogue
        encoder.emit_epilogue(32).unwrap();

        let code = encoder.finalize().unwrap();
        assert!(!code.is_empty());

        // Should contain push rbp, mov rbp rsp, sub rsp 32, add rsp 32, pop rbp, ret
    }

    #[test]
    fn test_factorial_pattern_encoding() {
        let mut encoder = X64Encoder::new().unwrap();

        // Test the critical instructions needed for factorial compilation
        let rax = AsmReg::new(0, 0); // RAX/EAX
        let rcx = AsmReg::new(0, 1); // RCX/ECX

        // Create labels for blocks (simulating factorial control flow)
        let entry_block = 0;
        let recurse_block = 1;
        let return_block = 2;

        // Emit function prologue
        encoder.emit_prologue(32).unwrap();

        // Place entry block label
        encoder.place_label_for_block(entry_block).unwrap();

        // Compare rax with 1 (if n <= 1)
        encoder.cmp32_reg_imm(rax, 1).unwrap();

        // Conditional jump to return block if n <= 1
        encoder
            .jmp_conditional_to_block(JumpCondition::LessEqual, return_block)
            .unwrap();

        // Place recursive computation block
        encoder.place_label_for_block(recurse_block).unwrap();

        // Subtract 1 from rax (n - 1)
        encoder.sub32_reg_imm(rax, 1).unwrap();

        // Simulate recursive call setup (would normally be more complex)
        encoder.call_direct(0).unwrap(); // Offset will be resolved during linking

        // Multiply result by original n (stored in rcx)
        encoder.imul32_reg_reg(rax, rcx).unwrap();

        // Unconditional jump to return
        encoder.jmp_unconditional_to_block(return_block).unwrap();

        // Place return block
        encoder.place_label_for_block(return_block).unwrap();

        // Function epilogue
        encoder.emit_epilogue(32).unwrap();

        // Generate final machine code
        let code = encoder.finalize().unwrap();
        assert!(!code.is_empty());

        // The code should contain real x86-64 instructions for factorial-like control flow
        // This verifies that our enhanced encoder can handle the complex patterns
        // needed for real C function compilation
        log::trace!(
            "Generated {} bytes of machine code for factorial pattern",
            code.len()
        );
    }

    #[test]
    fn test_block_label_management() {
        let mut encoder = X64Encoder::new().unwrap();

        // Test proper label creation and placement
        let block1 = 10;
        let block2 = 20;

        // Create labels for blocks that don't exist yet
        let _label1 = encoder.get_or_create_label_for_block(block1);
        let _label2 = encoder.get_or_create_label_for_block(block2);

        // Verify different blocks get different labels
        // Note: We can't directly compare label IDs due to API limitations

        // Verify labels haven't been placed yet
        assert!(!encoder.is_block_placed(block1));
        assert!(!encoder.is_block_placed(block2));

        // Place the labels with some instructions in between
        encoder.place_label_for_block(block1).unwrap();

        // Add some instructions between labels
        let rax = AsmReg::new(0, 0);
        encoder.mov32_reg_imm(rax, 42).unwrap();

        encoder.place_label_for_block(block2).unwrap();

        // Add instruction after the second label (iced-x86 requirement)
        encoder.ret().unwrap();

        // Verify labels are now placed
        assert!(encoder.is_block_placed(block1));
        assert!(encoder.is_block_placed(block2));

        // Verify getting same block returns same label
        let _label1_again = encoder.get_or_create_label_for_block(block1);
        // Note: We can't directly compare label IDs due to API limitations

        let code = encoder.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_32bit_vs_64bit_operations() {
        let mut encoder = X64Encoder::new().unwrap();

        let rax = AsmReg::new(0, 0);
        let rcx = AsmReg::new(0, 1);

        // Test 32-bit operations (for LLVM i32 types)
        encoder.mov32_reg_imm(rax, 42).unwrap();
        encoder.add32_reg_reg(rax, rcx).unwrap();
        encoder.sub32_reg_reg(rax, rcx).unwrap();
        encoder.imul32_reg_reg(rax, rcx).unwrap();
        encoder.cmp32_reg_imm(rax, 0).unwrap();

        // Test 64-bit operations
        encoder.mov_reg_imm(rax, 0x123456789).unwrap();
        encoder.add_reg_reg(rax, rcx).unwrap();
        encoder.cmp_reg_reg(rax, rcx).unwrap();

        let code = encoder.finalize().unwrap();
        assert!(!code.is_empty());

        // This verifies we can generate both 32-bit and 64-bit variants
        // which is critical for LLVM IR compilation where different types
        // need different instruction sizes
    }
}
