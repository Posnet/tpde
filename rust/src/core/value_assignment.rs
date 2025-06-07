// This module implements the ValueAssignment system for tracking where IR values are stored
// (registers, stack, or memory) during compilation. It manages multi-part values (e.g., 128-bit
// values split into two 64-bit parts), reference counting for lifetime management, and storage
// allocation. Key components: PartData (packed 16-bit representation of register assignments),
// ValueAssignment (tracks location and parts for each value), AssignmentFlags (control behavior
// like delay_free and variable_ref), StorageLocation (stack offset or variable reference),
// AssignmentAllocator (efficient memory pool with free lists by part count), and
// ValueAssignmentManager (maps local value indices to assignments). The system supports
// efficient register allocation by tracking which values are in registers vs spilled to stack,
// manages reference counts to know when storage can be freed, and handles complex values that
// span multiple registers. This is a core component enabling TPDE's fast register allocation.

//! Value assignment system for tracking register and stack locations.
//!
//! This module implements the core ValueAssignment infrastructure that tracks
//! where IR values are stored (registers vs stack) and manages the allocation
//! and lifetime of these storage locations. This is the foundation of TPDE's
//! register allocation system.

use std::collections::HashMap;

/// Index type for local values within a function.
pub type ValLocalIdx = usize;

/// Flags controlling ValueAssignment behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssignmentFlags {
    /// Whether the assignment is a variable reference (global, alloca).
    pub variable_ref: bool,
    /// Whether the variable reference refers to a stack slot.
    pub stack_variable: bool,
    /// Whether to delay freeing when reference count reaches zero.
    pub delay_free: bool,
    /// Whether the assignment is in the delayed free list.
    pub pending_free: bool,
}

/// Storage location for a value assignment.
#[derive(Debug, Clone, Copy)]
pub enum StorageLocation {
    /// Value stored on stack at given frame offset.
    Stack(i32),
    /// Variable reference with custom index.
    VarRef(u32),
}

/// Individual part of a multi-part value assignment.
///
/// Values can be split across multiple parts (e.g., 128-bit value as two 64-bit parts).
/// Each part tracks its register assignment and state independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartData {
    /// Packed representation: [reg_bank:3][reg_id:5][size_log2:3][flags:5]
    packed: u16,
}

impl PartData {
    /// Create new part data for the given register bank, ID, and size.
    pub fn new(reg_bank: u8, reg_id: u8, size_log2: u8) -> Self {
        assert!(reg_bank < 8, "Register bank must fit in 3 bits");
        assert!(reg_id < 32, "Register ID must fit in 5 bits");
        assert!(size_log2 < 8, "Size log2 must fit in 3 bits");

        Self {
            packed: ((reg_bank as u16) << 13) | ((reg_id as u16) << 8) | ((size_log2 as u16) << 5),
        }
    }

    /// Get the register bank for this part.
    pub fn reg_bank(&self) -> u8 {
        ((self.packed >> 13) & 0x7) as u8
    }

    /// Get the register ID for this part.
    pub fn reg_id(&self) -> u8 {
        ((self.packed >> 8) & 0x1F) as u8
    }

    /// Get the size of this part as log2 (0=1byte, 1=2bytes, 2=4bytes, etc.).
    pub fn size_log2(&self) -> u8 {
        ((self.packed >> 5) & 0x7) as u8
    }

    /// Get the size of this part in bytes.
    pub fn size_bytes(&self) -> u32 {
        1u32 << self.size_log2()
    }

    /// Whether register assignment is valid.
    pub fn register_valid(&self) -> bool {
        (self.packed & 0x1) != 0
    }

    /// Set register validity flag.
    pub fn set_register_valid(&mut self, valid: bool) {
        self.packed = if valid { self.packed | 0x1 } else { self.packed & !0x1 };
    }

    /// Whether the part has been modified since last spill.
    pub fn is_modified(&self) -> bool {
        (self.packed & 0x2) != 0
    }

    /// Set modification flag.
    pub fn set_modified(&mut self, modified: bool) {
        self.packed = if modified { self.packed | 0x2 } else { self.packed & !0x2 };
    }
}

/// Core value assignment tracking register and stack locations.
///
/// Each ValueAssignment represents where an IR value is stored and manages
/// the lifecycle of that storage. Values can be split into multiple parts
/// for efficient handling of large or complex types.
#[derive(Debug)]
pub struct ValueAssignment {
    /// Storage location (stack offset or variable reference index).
    pub location: Option<StorageLocation>,
    /// Number of parts this value is split into.
    pub part_count: u32,
    /// Maximum size of any part in bytes.
    pub max_part_size: u8,
    /// Number of active references to this assignment.
    pub references_left: u32,
    /// Control flags for assignment behavior.
    pub flags: AssignmentFlags,
    /// Per-part register assignment and state.
    pub parts: Vec<PartData>,
}

impl ValueAssignment {
    /// Create a new value assignment with the given number of parts.
    pub fn new(part_count: u32, max_part_size: u8) -> Self {
        Self {
            location: None,
            part_count,
            max_part_size,
            references_left: 0,
            flags: AssignmentFlags::default(),
            parts: vec![PartData::new(0, 0, 0); part_count as usize],
        }
    }

    /// Create a stack-based assignment at the given frame offset.
    pub fn new_stack(frame_offset: i32, part_count: u32, max_part_size: u8) -> Self {
        let mut assignment = Self::new(part_count, max_part_size);
        assignment.location = Some(StorageLocation::Stack(frame_offset));
        assignment
    }

    /// Create a variable reference assignment.
    pub fn new_var_ref(var_idx: u32, stack_variable: bool) -> Self {
        let mut assignment = Self::new(1, 8); // Single-part, pointer-sized
        assignment.location = Some(StorageLocation::VarRef(var_idx));
        assignment.flags.variable_ref = true;
        assignment.flags.stack_variable = stack_variable;
        assignment
    }

    /// Get the total size of this assignment in bytes.
    pub fn size(&self) -> u32 {
        assert!(
            !self.flags.variable_ref,
            "Variable references have no allocation size"
        );
        self.part_count * self.max_part_size as u32
    }

    /// Get a specific part of this assignment.
    pub fn part(&self, index: u32) -> Option<&PartData> {
        self.parts.get(index as usize)
    }

    /// Get a mutable reference to a specific part.
    pub fn part_mut(&mut self, index: u32) -> Option<&mut PartData> {
        self.parts.get_mut(index as usize)
    }

    /// Increment reference count.
    pub fn add_ref(&mut self) {
        self.references_left += 1;
    }

    /// Decrement reference count, returning true if assignment should be freed.
    pub fn remove_ref(&mut self) -> bool {
        assert!(self.references_left > 0, "Reference count underflow");
        self.references_left -= 1;
        self.references_left == 0 && !self.flags.delay_free
    }

    /// Check if assignment is ready for delayed freeing.
    pub fn should_delay_free(&self) -> bool {
        self.references_left == 0 && self.flags.delay_free
    }
}

/// Allocator for ValueAssignment objects with efficient memory management.
///
/// Uses a combination of free lists and bump allocation for optimal performance
/// in the common case of single-part assignments.
#[derive(Debug, Default)]
pub struct AssignmentAllocator {
    /// Storage pool for assignments.
    assignments: Vec<ValueAssignment>,
    /// Free list indices by part count (up to 4 parts, then use general list).
    free_lists: [Vec<usize>; 5],
    /// Free list for assignments with >4 parts.
    large_free_list: Vec<usize>,
}

impl AssignmentAllocator {
    /// Create a new assignment allocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new assignment with the given number of parts.
    pub fn allocate(&mut self, part_count: u32, max_part_size: u8) -> usize {
        let free_list = if part_count <= 4 {
            &mut self.free_lists[part_count as usize - 1]
        } else {
            &mut self.large_free_list
        };

        if let Some(idx) = free_list.pop() {
            // Reuse existing assignment
            let assignment = &mut self.assignments[idx];
            *assignment = ValueAssignment::new(part_count, max_part_size);
            idx
        } else {
            // Allocate new assignment
            let idx = self.assignments.len();
            self.assignments
                .push(ValueAssignment::new(part_count, max_part_size));
            idx
        }
    }

    /// Free an assignment back to the allocator.
    pub fn free(&mut self, idx: usize) {
        if idx >= self.assignments.len() {
            return; // Invalid index
        }

        let assignment = &mut self.assignments[idx];
        let part_count = assignment.part_count;

        // Reset assignment state
        assignment.references_left = 0;
        assignment.flags.pending_free = false;

        // Add to appropriate free list
        let free_list = if part_count <= 4 {
            &mut self.free_lists[part_count as usize - 1]
        } else {
            &mut self.large_free_list
        };
        free_list.push(idx);
    }

    /// Get a reference to an assignment.
    pub fn get(&self, idx: usize) -> Option<&ValueAssignment> {
        self.assignments.get(idx)
    }

    /// Get a mutable reference to an assignment.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut ValueAssignment> {
        self.assignments.get_mut(idx)
    }
}

/// Manager for all value assignments within a function.
///
/// Tracks assignments for each local value and provides the interface
/// for register allocation and stack management.
#[derive(Debug)]
pub struct ValueAssignmentManager {
    /// Assignment allocator for memory management.
    allocator: AssignmentAllocator,
    /// Mapping from local value index to assignment index.
    assignments: HashMap<ValLocalIdx, usize>,
    /// List of assignments pending delayed freeing.
    delayed_free_list: Vec<usize>,
}

impl Default for ValueAssignmentManager {
    fn default() -> Self {
        Self {
            allocator: AssignmentAllocator::new(),
            assignments: HashMap::new(),
            delayed_free_list: Vec::new(),
        }
    }
}

impl ValueAssignmentManager {
    /// Create a new value assignment manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an assignment for the given local value.
    pub fn create_assignment(
        &mut self,
        local_idx: ValLocalIdx,
        part_count: u32,
        max_part_size: u8,
    ) -> &mut ValueAssignment {
        let assignment_idx = self.allocator.allocate(part_count, max_part_size);
        self.assignments.insert(local_idx, assignment_idx);
        self.allocator.get_mut(assignment_idx).unwrap()
    }

    /// Get the assignment for a local value.
    pub fn get_assignment(&self, local_idx: ValLocalIdx) -> Option<&ValueAssignment> {
        self.assignments
            .get(&local_idx)
            .and_then(|&idx| self.allocator.get(idx))
    }

    /// Get a mutable reference to the assignment for a local value.
    pub fn get_assignment_mut(&mut self, local_idx: ValLocalIdx) -> Option<&mut ValueAssignment> {
        self.assignments
            .get(&local_idx)
            .and_then(|&idx| self.allocator.get_mut(idx))
    }

    /// Remove reference to a value, potentially freeing its assignment.
    pub fn remove_ref(&mut self, local_idx: ValLocalIdx) {
        #[allow(clippy::collapsible_if)]
        if let Some(&assignment_idx) = self.assignments.get(&local_idx) {
            if let Some(assignment) = self.allocator.get_mut(assignment_idx) {
                if assignment.remove_ref() {
                    // Assignment is ready to be freed
                    self.assignments.remove(&local_idx);
                    self.allocator.free(assignment_idx);
                } else if assignment.should_delay_free() {
                    // Add to delayed free list
                    assignment.flags.pending_free = true;
                    self.delayed_free_list.push(assignment_idx);
                }
            }
        }
    }

    /// Process delayed free list, freeing assignments that are no longer live.
    pub fn process_delayed_free(&mut self) {
        let mut remaining = Vec::new();

        for assignment_idx in self.delayed_free_list.drain(..) {
            if let Some(assignment) = self.allocator.get_mut(assignment_idx) {
                if assignment.references_left == 0 {
                    // Free the assignment
                    if let Some((&local_idx, _)) = self
                        .assignments
                        .iter()
                        .find(|&(_, &idx)| idx == assignment_idx)
                    {
                        self.assignments.remove(&local_idx);
                    }
                    self.allocator.free(assignment_idx);
                } else {
                    // Still has references, keep in delayed list
                    remaining.push(assignment_idx);
                }
            }
        }

        self.delayed_free_list = remaining;
    }

    /// Reset the manager for a new function.
    pub fn reset(&mut self) {
        self.assignments.clear();
        self.delayed_free_list.clear();
        // Note: allocator keeps its memory pool for reuse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part_data_packing() {
        let part = PartData::new(3, 15, 2); // bank 3, reg 15, size 4 bytes
        assert_eq!(part.reg_bank(), 3);
        assert_eq!(part.reg_id(), 15);
        assert_eq!(part.size_log2(), 2);
        assert_eq!(part.size_bytes(), 4);
        assert!(!part.register_valid());
        assert!(!part.is_modified());
    }

    #[test]
    fn test_part_data_flags() {
        let mut part = PartData::new(0, 0, 0);

        part.set_register_valid(true);
        assert!(part.register_valid());

        part.set_modified(true);
        assert!(part.is_modified());

        part.set_register_valid(false);
        assert!(!part.register_valid());
        assert!(part.is_modified()); // Should still be modified
    }

    #[test]
    fn test_value_assignment_creation() {
        let assignment = ValueAssignment::new(2, 8);
        assert_eq!(assignment.part_count, 2);
        assert_eq!(assignment.max_part_size, 8);
        assert_eq!(assignment.size(), 16);
        assert_eq!(assignment.parts.len(), 2);
    }

    #[test]
    fn test_stack_assignment() {
        let assignment = ValueAssignment::new_stack(-16, 1, 4);
        match assignment.location {
            Some(StorageLocation::Stack(offset)) => assert_eq!(offset, -16),
            _ => panic!("Expected stack location"),
        }
    }

    #[test]
    fn test_reference_counting() {
        let mut assignment = ValueAssignment::new(1, 4);
        assignment.add_ref();
        assignment.add_ref();
        assert_eq!(assignment.references_left, 2);

        assert!(!assignment.remove_ref());
        assert_eq!(assignment.references_left, 1);

        assert!(assignment.remove_ref());
        assert_eq!(assignment.references_left, 0);
    }

    #[test]
    fn test_assignment_manager() {
        let mut manager = ValueAssignmentManager::new();

        let assignment = manager.create_assignment(0, 1, 8);
        assignment.add_ref();

        assert!(manager.get_assignment(0).is_some());
        assert_eq!(manager.get_assignment(0).unwrap().part_count, 1);

        manager.remove_ref(0);
        assert!(manager.get_assignment(0).is_none());
    }
}
