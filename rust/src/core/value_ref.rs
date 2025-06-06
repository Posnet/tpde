//! RAII wrappers for safe register allocation and value management.
//!
//! This module provides ValueRef and ValuePartRef - the high-level interfaces
//! that instruction selection code uses to interact with the register allocator.
//! These provide automatic resource management, reference counting, and safe
//! access to registers while abstracting the complexity of the underlying
//! ValueAssignment and RegisterFile systems.

use super::value_assignment::{ValLocalIdx, ValueAssignmentManager};
use super::register_file::{RegisterFile, AsmReg, RegAllocError};

/// Error types for value reference operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueRefError {
    /// Value not found in assignment manager.
    ValueNotFound,
    /// Register allocation failed.
    RegisterAllocationFailed(RegAllocError),
    /// Attempted to access invalid part of multi-part value.
    InvalidPart,
    /// Value is not currently in a register.
    NotInRegister,
    /// Operation not supported for this value type.
    UnsupportedOperation,
}

/// Ownership mode for ValueRef RAII semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipMode {
    /// Unowned reference - no cleanup responsibility.
    Unowned,
    /// Reference counted - participates in shared ownership.
    RefCounted,
    /// Owned - exclusive ownership, responsible for cleanup.
    Owned,
}

/// Execution context providing access to register allocation infrastructure.
///
/// This allows ValueRef and ValuePartRef to access the necessary systems
/// without holding direct references, avoiding Rust borrowing conflicts.
pub struct CompilerContext<'a> {
    pub assignments: &'a mut ValueAssignmentManager,
    pub register_file: &'a mut RegisterFile,
}

impl<'a> CompilerContext<'a> {
    pub fn new(
        assignments: &'a mut ValueAssignmentManager,
        register_file: &'a mut RegisterFile,
    ) -> Self {
        Self {
            assignments,
            register_file,
        }
    }
}

/// RAII wrapper for value references with automatic register management.
///
/// ValueRef provides the primary interface for instruction selection code
/// to interact with values. It uses a context pattern to access the underlying
/// register allocation systems without holding references directly.
pub struct ValueRef {
    /// Local index of the value within the function.
    local_idx: ValLocalIdx,
    /// Ownership mode determining cleanup responsibility.
    ownership: OwnershipMode,
    /// Whether this ValueRef has been consumed (moved).
    consumed: bool,
}

impl ValueRef {
    /// Create a new ValueRef with the specified ownership mode.
    pub fn new(
        local_idx: ValLocalIdx,
        ownership: OwnershipMode,
        ctx: &mut CompilerContext,
    ) -> Self {
        // Add reference count if using ref-counted ownership
        if ownership == OwnershipMode::RefCounted {
            if let Some(assignment) = ctx.assignments.get_assignment_mut(local_idx) {
                assignment.add_ref();
            }
        }

        Self {
            local_idx,
            ownership,
            consumed: false,
        }
    }

    /// Create an owned ValueRef (takes exclusive ownership).
    pub fn owned(local_idx: ValLocalIdx, ctx: &mut CompilerContext) -> Self {
        // For owned values, we need to add a reference that we'll remove on drop
        if let Some(assignment) = ctx.assignments.get_assignment_mut(local_idx) {
            assignment.add_ref();
        }
        
        Self {
            local_idx,
            ownership: OwnershipMode::Owned,
            consumed: false,
        }
    }

    /// Create a ref-counted ValueRef (shared ownership).
    pub fn ref_counted(local_idx: ValLocalIdx, ctx: &mut CompilerContext) -> Self {
        Self::new(local_idx, OwnershipMode::RefCounted, ctx)
    }

    /// Get the local index of this value.
    pub fn local_idx(&self) -> ValLocalIdx {
        self.local_idx
    }

    /// Get the number of parts for this value.
    pub fn part_count(&self, ctx: &CompilerContext) -> Result<u32, ValueRefError> {
        ctx.assignments
            .get_assignment(self.local_idx)
            .map(|a| a.part_count)
            .ok_or(ValueRefError::ValueNotFound)
    }

    /// Access a specific part of this multi-part value.
    pub fn part(&self, part_idx: u32, ctx: &mut CompilerContext) -> Result<ValuePartRef, ValueRefError> {
        let part_count = self.part_count(ctx)?;
        if part_idx >= part_count {
            return Err(ValueRefError::InvalidPart);
        }

        ValuePartRef::new(self.local_idx, part_idx)
    }

    /// Access the first (and usually only) part of this value.
    pub fn single_part(&self, ctx: &mut CompilerContext) -> Result<ValuePartRef, ValueRefError> {
        self.part(0, ctx)
    }

    /// Check if this value is currently assigned to a register.
    pub fn is_in_register(&self, part_idx: u32, ctx: &CompilerContext) -> bool {
        if let Some(assignment) = ctx.assignments.get_assignment(self.local_idx) {
            if let Some(part_data) = assignment.part(part_idx) {
                return part_data.register_valid();
            }
        }
        false
    }

    /// Get the register for a specific part if it's currently assigned.
    pub fn current_register(&self, part_idx: u32, ctx: &CompilerContext) -> Option<AsmReg> {
        if !self.is_in_register(part_idx, ctx) {
            return None;
        }

        ctx.assignments
            .get_assignment(self.local_idx)
            .and_then(|assignment| assignment.part(part_idx))
            .map(|part_data| AsmReg::new(part_data.reg_bank(), part_data.reg_id()))
    }

    /// Mark this ValueRef as consumed (for move semantics).
    pub fn consume(&mut self) {
        self.consumed = true;
    }

    /// Check if this ValueRef has been consumed.
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    /// Manually drop this ValueRef with access to the context.
    pub fn drop_with_context(mut self, ctx: &mut CompilerContext) {
        self.cleanup(ctx);
        self.consumed = true; // Prevent double cleanup
    }

    /// Internal cleanup method.
    fn cleanup(&mut self, ctx: &mut CompilerContext) {
        if !self.consumed && self.ownership != OwnershipMode::Unowned {
            ctx.assignments.remove_ref(self.local_idx);
        }
    }
}

/// RAII wrapper for individual parts of multi-part values.
///
/// ValuePartRef provides fine-grained access to individual parts of values,
/// enabling efficient handling of large values that span multiple registers.
pub struct ValuePartRef {
    /// Local index of the parent value.
    local_idx: ValLocalIdx,
    /// Index of this part within the value.
    part_idx: u32,
    /// Currently locked register (if any).
    locked_register: Option<AsmReg>,
    /// Whether this part has been modified.
    modified: bool,
}

impl ValuePartRef {
    /// Create a new ValuePartRef for the specified value part.
    pub fn new(
        local_idx: ValLocalIdx,
        part_idx: u32,
    ) -> Result<Self, ValueRefError> {
        Ok(Self {
            local_idx,
            part_idx,
            locked_register: None,
            modified: false,
        })
    }

    /// Load this part into a register, allocating if necessary.
    ///
    /// This is the primary method used by instruction selection to get
    /// a register containing the value. It handles allocation, reloading
    /// from stack, and locking automatically.
    pub fn load_to_reg(&mut self, ctx: &mut CompilerContext) -> Result<AsmReg, ValueRefError> {
        // Check if already in a register
        if let Some(reg) = self.current_register(ctx) {
            // Lock the register to prevent eviction
            ctx.register_file.lock_register(reg)
                .map_err(ValueRefError::RegisterAllocationFailed)?;
            self.locked_register = Some(reg);
            return Ok(reg);
        }

        // Need to allocate a new register
        self.allocate_register(ctx)
    }

    /// Allocate a register for this part.
    fn allocate_register(&mut self, ctx: &mut CompilerContext) -> Result<AsmReg, ValueRefError> {
        let assignment = ctx.assignments
            .get_assignment(self.local_idx)
            .ok_or(ValueRefError::ValueNotFound)?;

        let part_data = assignment
            .part(self.part_idx)
            .ok_or(ValueRefError::InvalidPart)?;

        // Determine register bank based on value type/size
        let reg_bank = part_data.reg_bank();
        
        // Allocate register
        let reg = ctx.register_file
            .allocate_reg(reg_bank, self.local_idx, self.part_idx, None)
            .map_err(ValueRefError::RegisterAllocationFailed)?;

        // Update assignment with new register
        if let Some(assignment) = ctx.assignments.get_assignment_mut(self.local_idx) {
            if let Some(part_data) = assignment.part_mut(self.part_idx) {
                *part_data = crate::core::value_assignment::PartData::new(
                    reg.bank, 
                    reg.id, 
                    part_data.size_log2()
                );
                part_data.set_register_valid(true);
            }
        }

        // Lock the register
        ctx.register_file.lock_register(reg)
            .map_err(ValueRefError::RegisterAllocationFailed)?;
        self.locked_register = Some(reg);

        Ok(reg)
    }

    /// Try to reuse a register from another ValuePartRef.
    ///
    /// This enables efficient register reuse patterns in instruction selection.
    pub fn alloc_try_reuse(&mut self, other: &mut ValuePartRef, ctx: &mut CompilerContext) -> Result<AsmReg, ValueRefError> {
        if let Some(other_reg) = other.current_register(ctx) {
            // Check if we can take ownership of the other register
            if other.can_transfer_ownership(ctx) {
                // Transfer the register assignment
                other.release_register(ctx)?;
                self.claim_register(other_reg, ctx)?;
                return Ok(other_reg);
            }
        }

        // Fallback to normal allocation
        self.load_to_reg(ctx)
    }

    /// Get the current register for this part (if any).
    pub fn current_register(&self, ctx: &CompilerContext) -> Option<AsmReg> {
        ctx.assignments
            .get_assignment(self.local_idx)
            .and_then(|assignment| assignment.part(self.part_idx))
            .filter(|part_data| part_data.register_valid())
            .map(|part_data| AsmReg::new(part_data.reg_bank(), part_data.reg_id()))
    }

    /// Check if this part can transfer ownership of its register.
    fn can_transfer_ownership(&self, ctx: &CompilerContext) -> bool {
        // Can transfer if we have exclusive access and the register isn't shared
        self.locked_register.is_some() &&
        ctx.assignments.get_assignment(self.local_idx)
            .map(|a| a.references_left <= 1)
            .unwrap_or(false)
    }

    /// Release ownership of the current register.
    fn release_register(&mut self, ctx: &mut CompilerContext) -> Result<(), ValueRefError> {
        if let Some(reg) = self.locked_register.take() {
            // Unlock the register
            ctx.register_file.unlock_register(reg)
                .map_err(ValueRefError::RegisterAllocationFailed)?;

            // Mark register as invalid in assignment
            if let Some(assignment) = ctx.assignments.get_assignment_mut(self.local_idx) {
                if let Some(part_data) = assignment.part_mut(self.part_idx) {
                    part_data.set_register_valid(false);
                }
            }
        }
        Ok(())
    }

    /// Claim ownership of a register.
    fn claim_register(&mut self, reg: AsmReg, ctx: &mut CompilerContext) -> Result<(), ValueRefError> {
        // Update our assignment
        if let Some(assignment) = ctx.assignments.get_assignment_mut(self.local_idx) {
            if let Some(part_data) = assignment.part_mut(self.part_idx) {
                *part_data = crate::core::value_assignment::PartData::new(
                    reg.bank,
                    reg.id,
                    part_data.size_log2()
                );
                part_data.set_register_valid(true);
            }
        }

        // Lock the register
        ctx.register_file.lock_register(reg)
            .map_err(ValueRefError::RegisterAllocationFailed)?;
        self.locked_register = Some(reg);

        Ok(())
    }

    /// Mark this part as modified (dirty).
    pub fn set_modified(&mut self, ctx: &mut CompilerContext) {
        self.modified = true;
        if let Some(reg) = self.current_register(ctx) {
            ctx.register_file.mark_clobbered(reg);
        }
    }

    /// Check if this part has been modified.
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Spill this part to memory if it's in a register.
    pub fn spill(&mut self, ctx: &mut CompilerContext) -> Result<(), ValueRefError> {
        if let Some(_reg) = self.current_register(ctx) {
            // In a real implementation, this would generate spill code
            // For now, we just mark the register as invalid
            if let Some(assignment) = ctx.assignments.get_assignment_mut(self.local_idx) {
                if let Some(part_data) = assignment.part_mut(self.part_idx) {
                    part_data.set_register_valid(false);
                }
            }
        }
        Ok(())
    }

    /// Get the size of this part in bytes.
    pub fn size_bytes(&self, ctx: &CompilerContext) -> u32 {
        ctx.assignments
            .get_assignment(self.local_idx)
            .and_then(|assignment| assignment.part(self.part_idx))
            .map(|part_data| part_data.size_bytes())
            .unwrap_or(0)
    }

    /// Manually drop this ValuePartRef with access to the context.
    pub fn drop_with_context(mut self, ctx: &mut CompilerContext) {
        self.cleanup(ctx);
    }

    /// Internal cleanup method.
    fn cleanup(&mut self, ctx: &mut CompilerContext) {
        // Unlock any locked register
        if let Some(reg) = self.locked_register {
            let _ = ctx.register_file.unlock_register(reg);
        }
    }
}

/// Builder for creating ValueRef instances with proper ownership semantics.
///
/// This helps determine the correct ownership mode based on liveness analysis
/// and reference counting information.
pub struct ValueRefBuilder;

impl ValueRefBuilder {
    /// Create a ValueRef with ownership determined by liveness analysis.
    pub fn build_for_value(
        local_idx: ValLocalIdx,
        ctx: &mut CompilerContext,
        is_last_use: bool,
    ) -> ValueRef {
        let ownership = if is_last_use {
            // If this is the last use, take exclusive ownership
            OwnershipMode::Owned
        } else {
            // Otherwise use reference counting
            OwnershipMode::RefCounted
        };

        ValueRef::new(local_idx, ownership, ctx)
    }

    /// Create a ValueRef for a result value (always owned).
    pub fn build_for_result(
        local_idx: ValLocalIdx,
        ctx: &mut CompilerContext,
    ) -> ValueRef {
        ValueRef::owned(local_idx, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_setup() -> (ValueAssignmentManager, RegisterFile) {
        let assignments = ValueAssignmentManager::new();
        
        let mut allocatable = crate::core::register_file::RegBitSet::new();
        allocatable.union(&crate::core::register_file::RegBitSet::all_in_bank(0, 8));
        let register_file = RegisterFile::new(8, 1, allocatable);
        
        (assignments, register_file)
    }

    #[test]
    fn test_value_ref_creation() {
        let (mut assignments, mut register_file) = create_test_setup();
        let mut ctx = CompilerContext::new(&mut assignments, &mut register_file);
        
        // Create an assignment
        let assignment = ctx.assignments.create_assignment(42, 1, 8);
        assignment.add_ref();
        
        let value_ref = ValueRef::ref_counted(42, &mut ctx);
        assert_eq!(value_ref.local_idx(), 42);
        assert_eq!(value_ref.part_count(&ctx).unwrap(), 1);
        
        // Manual cleanup to test the context pattern
        value_ref.drop_with_context(&mut ctx);
    }

    #[test]
    fn test_value_part_ref_allocation() {
        let (mut assignments, mut register_file) = create_test_setup();
        let mut ctx = CompilerContext::new(&mut assignments, &mut register_file);
        
        // Create an assignment
        ctx.assignments.create_assignment(42, 1, 8);
        
        let value_ref = ValueRef::owned(42, &mut ctx);
        let mut part_ref = value_ref.single_part(&mut ctx).unwrap();
        
        // Load to register should succeed
        let reg = part_ref.load_to_reg(&mut ctx).unwrap();
        assert_eq!(reg.bank, 0);
        assert!(ctx.register_file.is_allocated(reg));
        assert!(ctx.register_file.is_locked(reg));
        
        // Manual cleanup
        part_ref.drop_with_context(&mut ctx);
        value_ref.drop_with_context(&mut ctx);
    }

    #[test]
    fn test_modification_tracking() {
        let (mut assignments, mut register_file) = create_test_setup();
        let mut ctx = CompilerContext::new(&mut assignments, &mut register_file);
        
        ctx.assignments.create_assignment(42, 1, 8);
        let value_ref = ValueRef::owned(42, &mut ctx);
        let mut part_ref = value_ref.single_part(&mut ctx).unwrap();
        
        assert!(!part_ref.is_modified());
        
        let reg = part_ref.load_to_reg(&mut ctx).unwrap();
        part_ref.set_modified(&mut ctx);
        
        assert!(part_ref.is_modified());
        assert!(ctx.register_file.is_clobbered(reg));
        
        // Manual cleanup
        part_ref.drop_with_context(&mut ctx);
        value_ref.drop_with_context(&mut ctx);
    }
}