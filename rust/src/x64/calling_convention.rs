// This module implements the System V AMD64 ABI calling convention for x86-64 Linux/Unix systems.
// It provides CCAssigner trait and SysVAssigner implementation to handle argument passing and
// return value placement according to the standard ABI. Key features: GP arguments pass in
// RDI, RSI, RDX, RCX, R8, R9 (first 6), FP arguments pass in XMM0-XMM7 (first 8), remaining
// arguments go on stack with 8-byte alignment, return values use RAX/RDX (int) or XMM0/XMM1
// (float), and callee-saved registers are RBX, R12-R15. FunctionFrame manages the stack layout
// including saved registers, spill slots, and local variables. The implementation ensures 16-byte
// stack alignment per ABI requirements and supports varargs functions via must_assign_stack flag.
// This mirrors the C++ CCAssigner infrastructure while adapting to Rust's type system and
// provides the foundation for correct C function interoperability.

//! System V x86-64 calling convention implementation.
//!
//! This module implements the System V ABI for x86-64, providing argument passing,
//! return value handling, and callee-saved register management. It closely mirrors
//! the C++ CCAssigner infrastructure while adapting to Rust's type system.

use crate::core::register_file::{AsmReg, RegBitSet};
use bumpalo::{collections::Vec as BumpVec, Bump};

/// Register banks for different register types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegBank {
    GeneralPurpose = 0,
    Xmm = 1,
}

/// Argument attributes for calling convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgAttribute {
    /// No special attributes.
    None,
    /// Pass by value (copy entire struct).
    ByVal { size: u32, align: u32 },
    /// Struct return (hidden first parameter).
    StructRet,
    /// Sign extend the value.
    SignExt,
    /// Zero extend the value.
    ZeroExt,
}

/// Argument assignment result from calling convention analysis.
#[derive(Debug, Clone, Copy)]
pub struct CCAssignment {
    /// Register bank this argument belongs to.
    pub bank: RegBank,
    /// Size of the argument in bytes.
    pub size: u32,
    /// Required alignment in bytes.
    pub align: u32,
    /// Number of consecutive registers needed.
    pub consecutive: u32,
    /// Assigned register (if any).
    pub reg: Option<AsmReg>,
    /// Stack offset (if assigned to stack).
    pub stack_off: Option<i32>,
    /// Argument attributes.
    pub attribute: ArgAttribute,
}

impl CCAssignment {
    /// Create a new assignment for an argument.
    pub fn new(bank: RegBank, size: u32, align: u32) -> Self {
        Self {
            bank,
            size,
            align,
            consecutive: 1,
            reg: None,
            stack_off: None,
            attribute: ArgAttribute::None,
        }
    }

    /// Create a new assignment with attribute.
    pub fn with_attribute(bank: RegBank, size: u32, align: u32, attribute: ArgAttribute) -> Self {
        Self {
            bank,
            size,
            align,
            consecutive: 1,
            reg: None,
            stack_off: None,
            attribute,
        }
    }
}

/// Information about a calling convention.
#[derive(Debug, Clone)]
pub struct CCInfo {
    /// Registers available for allocation.
    pub allocatable_regs: RegBitSet,
    /// Registers that must be preserved by callee.
    pub callee_saved_regs: RegBitSet,
    /// Registers used for argument passing.
    pub arg_regs: RegBitSet,
}

/// Trait for calling convention argument assignment.
pub trait CCAssigner {
    /// Get calling convention information.
    fn get_ccinfo(&self) -> &CCInfo;

    /// Assign an argument according to the calling convention.
    fn assign_arg(&mut self, arg: &mut CCAssignment);

    /// Assign a return value according to the calling convention.
    fn assign_ret(&mut self, arg: &mut CCAssignment);

    /// Reset state for a new function.
    fn reset(&mut self);

    /// Get the total stack space needed for arguments.
    fn get_stack_size(&self) -> u32;
}

/// System V x86-64 calling convention assigner.
///
/// Implements the standard Unix/Linux x86-64 ABI:
/// - First 6 integer args in RDI, RSI, RDX, RCX, R8, R9
/// - First 8 floating-point args in XMM0-XMM7
/// - Remaining args on stack
/// - Return values in RAX/RDX (int) or XMM0/XMM1 (float)
pub struct SysVAssigner {
    /// Calling convention info.
    cc_info: CCInfo,
    /// Current GP argument register index.
    gp_cnt: usize,
    /// Current XMM argument register index.
    xmm_cnt: usize,
    /// Current stack offset for arguments.
    stack: u32,
    /// Force remaining args to stack (for varargs).
    must_assign_stack: bool,
    /// Return value GP register count.
    ret_gp_cnt: usize,
    /// Return value XMM register count.
    ret_xmm_cnt: usize,
}

impl Default for SysVAssigner {
    fn default() -> Self {
        // Build allocatable register set (all except RSP, RBP)
        let mut allocatable = RegBitSet::new();
        for i in 0..16 {
            if i != 4 && i != 5 {
                // Skip RSP(4) and RBP(5)
                allocatable.set(AsmReg::new(0, i));
            }
        }
        // Add XMM registers
        for i in 0..16 {
            allocatable.set(AsmReg::new(1, i));
        }

        // Build callee-saved register set
        let mut callee_saved = RegBitSet::new();
        let callee_saved_ids = [3, 12, 13, 14, 15]; // RBX, R12-R15
        for &id in &callee_saved_ids {
            callee_saved.set(AsmReg::new(0, id));
        }

        // Build argument register set
        let mut arg_regs = RegBitSet::new();
        for &reg in &Self::GP_ARG_REGS {
            arg_regs.set(reg);
        }
        for &reg in &Self::XMM_ARG_REGS {
            arg_regs.set(reg);
        }

        let cc_info = CCInfo {
            allocatable_regs: allocatable,
            callee_saved_regs: callee_saved,
            arg_regs,
        };

        Self {
            cc_info,
            gp_cnt: 0,
            xmm_cnt: 0,
            stack: 0,
            must_assign_stack: false,
            ret_gp_cnt: 0,
            ret_xmm_cnt: 0,
        }
    }
}

impl SysVAssigner {
    /// System V x86-64 GP argument registers.
    const GP_ARG_REGS: [AsmReg; 6] = [
        AsmReg::new(0, 7), // RDI
        AsmReg::new(0, 6), // RSI
        AsmReg::new(0, 2), // RDX
        AsmReg::new(0, 1), // RCX
        AsmReg::new(0, 8), // R8
        AsmReg::new(0, 9), // R9
    ];

    /// System V x86-64 XMM argument registers.
    const XMM_ARG_REGS: [AsmReg; 8] = [
        AsmReg::new(1, 0), // XMM0
        AsmReg::new(1, 1), // XMM1
        AsmReg::new(1, 2), // XMM2
        AsmReg::new(1, 3), // XMM3
        AsmReg::new(1, 4), // XMM4
        AsmReg::new(1, 5), // XMM5
        AsmReg::new(1, 6), // XMM6
        AsmReg::new(1, 7), // XMM7
    ];

    /// Return value registers.
    const RET_GP_REGS: [AsmReg; 2] = [
        AsmReg::new(0, 0), // RAX
        AsmReg::new(0, 2), // RDX
    ];

    const RET_XMM_REGS: [AsmReg; 2] = [
        AsmReg::new(1, 0), // XMM0
        AsmReg::new(1, 1), // XMM1
    ];

    /// Create a new System V calling convention assigner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Force remaining arguments to be assigned to stack (for varargs).
    pub fn set_must_assign_stack(&mut self) {
        self.must_assign_stack = true;
    }

    /// Number of GP argument registers used so far.
    pub fn gp_used(&self) -> usize {
        self.gp_cnt
    }

    /// Number of XMM argument registers used so far.
    pub fn xmm_used(&self) -> usize {
        self.xmm_cnt
    }

    /// Align a value up to the specified alignment.
    fn align_up(value: u32, align: u32) -> u32 {
        (value + align - 1) & !(align - 1)
    }
}

impl CCAssigner for SysVAssigner {
    fn get_ccinfo(&self) -> &CCInfo {
        &self.cc_info
    }

    fn assign_arg(&mut self, arg: &mut CCAssignment) {
        // Handle special attributes
        match arg.attribute {
            ArgAttribute::ByVal { size, align } => {
                // byval arguments are always passed on stack as a copy
                let aligned_size = Self::align_up(size, 8);
                self.stack = Self::align_up(self.stack, align.max(8));
                arg.stack_off = Some(self.stack as i32);
                self.stack += aligned_size;
                return;
            }
            ArgAttribute::StructRet => {
                // sret is always the first argument and goes in RDI
                if self.gp_cnt == 0 {
                    arg.reg = Some(Self::GP_ARG_REGS[0]);
                    self.gp_cnt = 1;
                } else {
                    // This shouldn't happen - sret should be first arg
                    log::warn!("sret argument not first - assigning to stack");
                    self.stack = Self::align_up(self.stack, 8);
                    arg.stack_off = Some(self.stack as i32);
                    self.stack += 8;
                }
                return;
            }
            _ => {} // Handle normally
        }

        match arg.bank {
            RegBank::GeneralPurpose => {
                if !self.must_assign_stack
                    && self.gp_cnt + arg.consecutive as usize <= Self::GP_ARG_REGS.len()
                {
                    // Assign to GP register
                    arg.reg = Some(Self::GP_ARG_REGS[self.gp_cnt]);
                    self.gp_cnt += arg.consecutive as usize;
                } else {
                    // Assign to stack with proper alignment
                    let align = if arg.align < 8 { 8 } else { arg.align };
                    self.stack = Self::align_up(self.stack, align);
                    arg.stack_off = Some(self.stack as i32);
                    self.stack += 8; // Each stack slot is 8 bytes in System V
                }
            }
            RegBank::Xmm => {
                if !self.must_assign_stack
                    && self.xmm_cnt + arg.consecutive as usize <= Self::XMM_ARG_REGS.len()
                {
                    // Assign to XMM register
                    arg.reg = Some(Self::XMM_ARG_REGS[self.xmm_cnt]);
                    self.xmm_cnt += arg.consecutive as usize;
                } else {
                    // Assign to stack
                    let size = Self::align_up(arg.size, 8);
                    self.stack = Self::align_up(self.stack, size);
                    arg.stack_off = Some(self.stack as i32);
                    self.stack += size;
                }
            }
        }
    }

    fn assign_ret(&mut self, arg: &mut CCAssignment) {
        match arg.bank {
            RegBank::GeneralPurpose => {
                if self.ret_gp_cnt + arg.consecutive as usize <= Self::RET_GP_REGS.len() {
                    arg.reg = Some(Self::RET_GP_REGS[self.ret_gp_cnt]);
                    self.ret_gp_cnt += arg.consecutive as usize;
                }
            }
            RegBank::Xmm => {
                if self.ret_xmm_cnt + arg.consecutive as usize <= Self::RET_XMM_REGS.len() {
                    arg.reg = Some(Self::RET_XMM_REGS[self.ret_xmm_cnt]);
                    self.ret_xmm_cnt += arg.consecutive as usize;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.gp_cnt = 0;
        self.xmm_cnt = 0;
        self.stack = 0;
        self.must_assign_stack = false;
        self.ret_gp_cnt = 0;
        self.ret_xmm_cnt = 0;
    }

    fn get_stack_size(&self) -> u32 {
        // Align stack to 16-byte boundary as required by System V ABI
        Self::align_up(self.stack, 16)
    }
}

/// Function calling state and stack frame management.
#[derive(Debug)]
pub struct FunctionFrame<'a> {
    /// Callee-saved registers that need to be preserved.
    pub saved_registers: BumpVec<'a, AsmReg>,
    /// Total size of the stack frame.
    pub frame_size: u32,
    /// Offset to local variables area.
    pub locals_offset: i32,
    /// Stack slot assignments for spilled values.
    pub spill_slots: BumpVec<'a, i32>,
    /// Current spill slot offset.
    pub spill_offset: i32,
    /// Argument assignments for the current function.
    pub arg_assignments: BumpVec<'a, CCAssignment>,
    /// Return value assignments.
    pub ret_assignments: BumpVec<'a, CCAssignment>,
}

impl<'a> FunctionFrame<'a> {
    /// Create a new function frame using the provided arena.
    pub fn new(arena: &'a Bump) -> Self {
        Self {
            saved_registers: BumpVec::new_in(arena),
            frame_size: 0,
            locals_offset: 0,
            spill_slots: BumpVec::new_in(arena),
            spill_offset: -16, // Start below saved RBP
            arg_assignments: BumpVec::new_in(arena),
            ret_assignments: BumpVec::new_in(arena),
        }
    }

    /// Add a callee-saved register that needs preservation.
    pub fn add_saved_register(&mut self, reg: AsmReg) {
        if !self.saved_registers.contains(&reg) {
            self.saved_registers.push(reg);
            self.spill_offset -= 8; // Each saved register takes 8 bytes
        }
    }

    /// Allocate a new spill slot and return its offset.
    pub fn allocate_spill_slot(&mut self, size: u32) -> i32 {
        let aligned_size = size.div_ceil(8) * 8; // Align to 8 bytes
        self.spill_offset -= aligned_size as i32;
        let offset = self.spill_offset;
        self.spill_slots.push(offset);
        offset
    }

    /// Calculate the final frame size.
    pub fn calculate_frame_size(&mut self) {
        // Frame layout:
        // rbp + 8:  return address
        // rbp:      saved rbp
        // rbp - 8:  saved registers (callee-saved)
        // rbp - X:  local variables and spill slots
        // rbp - Y:  outgoing call arguments

        let saved_reg_size = self.saved_registers.len() as u32 * 8;
        let spill_size = if self.spill_offset < -16 {
            (-self.spill_offset - 16) as u32
        } else {
            0
        };

        // Total frame size must be 16-byte aligned for System V ABI
        let total_size = saved_reg_size + spill_size;
        self.frame_size = total_size.div_ceil(16) * 16;
        self.locals_offset = -(self.frame_size as i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn test_sysv_gp_argument_assignment() {
        let mut assigner = SysVAssigner::new();

        // Test first 6 GP arguments go to registers
        for i in 0..6 {
            let mut arg = CCAssignment::new(RegBank::GeneralPurpose, 8, 8);
            assigner.assign_arg(&mut arg);

            assert!(arg.reg.is_some());
            assert!(arg.stack_off.is_none());
            assert_eq!(arg.reg.unwrap(), SysVAssigner::GP_ARG_REGS[i]);
        }

        // 7th argument should go to stack
        let mut arg7 = CCAssignment::new(RegBank::GeneralPurpose, 8, 8);
        assigner.assign_arg(&mut arg7);

        assert!(arg7.reg.is_none());
        assert!(arg7.stack_off.is_some());
        assert_eq!(arg7.stack_off.unwrap(), 0);
    }

    #[test]
    fn test_sysv_xmm_argument_assignment() {
        let mut assigner = SysVAssigner::new();

        // Test first 8 XMM arguments go to registers
        for i in 0..8 {
            let mut arg = CCAssignment::new(RegBank::Xmm, 8, 8);
            assigner.assign_arg(&mut arg);

            assert!(arg.reg.is_some());
            assert!(arg.stack_off.is_none());
            assert_eq!(arg.reg.unwrap(), SysVAssigner::XMM_ARG_REGS[i]);
        }

        // 9th XMM argument should go to stack
        let mut arg9 = CCAssignment::new(RegBank::Xmm, 8, 8);
        assigner.assign_arg(&mut arg9);

        assert!(arg9.reg.is_none());
        assert!(arg9.stack_off.is_some());
    }

    #[test]
    fn test_sysv_return_value_assignment() {
        let mut assigner = SysVAssigner::new();

        // Test GP return value
        let mut ret_gp = CCAssignment::new(RegBank::GeneralPurpose, 8, 8);
        assigner.assign_ret(&mut ret_gp);

        assert!(ret_gp.reg.is_some());
        assert_eq!(ret_gp.reg.unwrap(), SysVAssigner::RET_GP_REGS[0]); // RAX

        // Test XMM return value
        let mut ret_xmm = CCAssignment::new(RegBank::Xmm, 8, 8);
        assigner.assign_ret(&mut ret_xmm);

        assert!(ret_xmm.reg.is_some());
        assert_eq!(ret_xmm.reg.unwrap(), SysVAssigner::RET_XMM_REGS[0]); // XMM0
    }

    #[test]
    fn test_must_assign_stack() {
        let mut assigner = SysVAssigner::new();
        assigner.set_must_assign_stack();

        // Even first argument should go to stack when must_assign_stack is set
        let mut arg = CCAssignment::new(RegBank::GeneralPurpose, 8, 8);
        assigner.assign_arg(&mut arg);

        assert!(arg.reg.is_none());
        assert!(arg.stack_off.is_some());
    }

    #[test]
    fn test_function_frame_spill_allocation() {
        let arena = Bump::new();
        let mut frame = FunctionFrame::new(&arena);

        // Allocate some spill slots
        let slot1 = frame.allocate_spill_slot(8);
        let slot2 = frame.allocate_spill_slot(4);
        let slot3 = frame.allocate_spill_slot(16);

        // Slots should be allocated in descending order
        assert!(slot1 > slot2);
        assert!(slot2 > slot3);

        // All slots should be 8-byte aligned
        assert_eq!(slot1 % 8, 0);
        assert_eq!(slot2 % 8, 0);
        assert_eq!(slot3 % 8, 0);
    }

    #[test]
    fn test_function_frame_saved_registers() {
        let arena = Bump::new();
        let mut frame = FunctionFrame::new(&arena);

        let rbx = AsmReg::new(0, 3);
        let r12 = AsmReg::new(0, 12);

        frame.add_saved_register(rbx);
        frame.add_saved_register(r12);
        frame.add_saved_register(rbx); // Should not duplicate

        assert_eq!(frame.saved_registers.len(), 2);
        assert!(frame.saved_registers.contains(&rbx));
        assert!(frame.saved_registers.contains(&r12));
    }
}
