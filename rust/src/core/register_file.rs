//! Register allocation and management.
//!
//! This module implements the RegisterFile system that performs actual register allocation,
//! tracks register usage, and manages eviction/spilling. This is the core of TPDE's
//! register allocation strategy, using a clock-based algorithm with on-demand spilling.

use super::value_assignment::ValLocalIdx;

/// Type alias for register spill callback function.
type SpillCallback = Box<dyn Fn(AsmReg, &Assignment) -> Result<(), String>>;

/// Maximum number of register banks supported (GP, FP, etc.).
pub const MAX_REGISTER_BANKS: usize = 4;

/// Maximum number of registers per bank.
pub const MAX_REGISTERS_PER_BANK: usize = 32;

/// Type for register bank indices.
pub type RegBank = u8;

/// Type for register IDs within a bank.
pub type RegId = u8;

/// Combined register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsmReg {
    pub bank: RegBank,
    pub id: RegId,
}

impl AsmReg {
    pub const fn new(bank: RegBank, id: RegId) -> Self {
        Self { bank, id }
    }

    /// Get the linear register index for array indexing.
    pub fn linear_index(&self, regs_per_bank: usize) -> usize {
        (self.bank as usize) * regs_per_bank + (self.id as usize)
    }

    /// Create from linear index.
    pub fn from_linear_index(index: usize, regs_per_bank: usize) -> Self {
        let bank = (index / regs_per_bank) as RegBank;
        let id = (index % regs_per_bank) as RegId;
        Self { bank, id }
    }
}

/// Bit set for efficiently tracking register sets.
#[derive(Debug, Clone)]
pub struct RegBitSet {
    /// Bit mask for each register bank.
    banks: [u64; MAX_REGISTER_BANKS],
}

impl RegBitSet {
    /// Create empty register set.
    pub fn new() -> Self {
        Self {
            banks: [0; MAX_REGISTER_BANKS],
        }
    }

    /// Create register set with all registers in bank marked.
    pub fn all_in_bank(bank: RegBank, count: u8) -> Self {
        let mut set = Self::new();
        if count <= 64 {
            set.banks[bank as usize] = (1u64 << count) - 1;
        }
        set
    }

    /// Check if register is set.
    pub fn contains(&self, reg: AsmReg) -> bool {
        if reg.bank as usize >= MAX_REGISTER_BANKS || reg.id >= 64 {
            return false;
        }
        (self.banks[reg.bank as usize] & (1u64 << reg.id)) != 0
    }

    /// Set a register.
    pub fn set(&mut self, reg: AsmReg) {
        if (reg.bank as usize) < MAX_REGISTER_BANKS && reg.id < 64 {
            self.banks[reg.bank as usize] |= 1u64 << reg.id;
        }
    }

    /// Clear a register.
    pub fn clear(&mut self, reg: AsmReg) {
        if (reg.bank as usize) < MAX_REGISTER_BANKS && reg.id < 64 {
            self.banks[reg.bank as usize] &= !(1u64 << reg.id);
        }
    }

    /// Set union with another set.
    pub fn union(&mut self, other: &RegBitSet) {
        for i in 0..MAX_REGISTER_BANKS {
            self.banks[i] |= other.banks[i];
        }
    }

    /// Set intersection with another set.
    pub fn intersect(&mut self, other: &RegBitSet) {
        for i in 0..MAX_REGISTER_BANKS {
            self.banks[i] &= other.banks[i];
        }
    }

    /// Find first set register in the given bank, excluding specified registers.
    pub fn find_first_in_bank(&self, bank: RegBank, exclude: &RegBitSet) -> Option<RegId> {
        if bank as usize >= MAX_REGISTER_BANKS {
            return None;
        }

        let available = self.banks[bank as usize] & !exclude.banks[bank as usize];
        if available == 0 {
            return None;
        }

        Some(available.trailing_zeros() as RegId)
    }

    /// Check if any registers are set in the given bank.
    pub fn any_in_bank(&self, bank: RegBank) -> bool {
        if bank as usize >= MAX_REGISTER_BANKS {
            return false;
        }
        self.banks[bank as usize] != 0
    }

    /// Count number of set registers in bank.
    pub fn count_in_bank(&self, bank: RegBank) -> u32 {
        if bank as usize >= MAX_REGISTER_BANKS {
            return 0;
        }
        self.banks[bank as usize].count_ones()
    }

    /// Clear all registers.
    pub fn clear_all(&mut self) {
        self.banks.fill(0);
    }
}

impl Default for RegBitSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Assignment tracking which value owns a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assignment {
    /// Local value index that owns this register.
    pub local_idx: ValLocalIdx,
    /// Which part of a multi-part value this register holds.
    pub part: u32,
}

impl Assignment {
    pub fn new(local_idx: ValLocalIdx, part: u32) -> Self {
        Self { local_idx, part }
    }
}

/// Error types for register allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegAllocError {
    /// No registers available in the requested bank.
    NoRegistersAvailable,
    /// Register is not allocated.
    RegisterNotAllocated,
    /// Invalid register bank or ID.
    InvalidRegister,
    /// Lock count underflow (too many unlocks).
    LockCountUnderflow,
}

/// RegisterFile manages register allocation for a single function.
///
/// Uses a clock-based allocation algorithm with reference counting and
/// on-demand spilling to efficiently manage register pressure.
pub struct RegisterFile {
    /// Number of registers per bank (must be ≤ 64).
    regs_per_bank: usize,
    /// Total number of registers across all banks.
    total_regs: usize,
    
    /// Registers available for allocation (excludes SP, BP, etc.).
    allocatable: RegBitSet,
    /// Currently allocated registers.
    used: RegBitSet,
    /// Registers that cannot be evicted (locked or fixed).
    fixed: RegBitSet,
    /// Registers that have been modified (for save/restore).
    clobbered: RegBitSet,
    
    /// Clock position for allocation in each bank.
    clocks: [RegId; MAX_REGISTER_BANKS],
    /// Which value owns each register.
    assignments: Vec<Option<Assignment>>,
    /// Lock count for each register (prevents eviction).
    lock_counts: Vec<u8>,
    
    /// Callback for spilling register contents.
    spill_callback: Option<SpillCallback>,
}

impl RegisterFile {
    /// Create a new register file with the given configuration.
    ///
    /// # Arguments
    /// * `regs_per_bank` - Number of registers per bank (max 64)
    /// * `num_banks` - Number of register banks (max 4)
    /// * `allocatable_regs` - Which registers are available for allocation
    pub fn new(
        regs_per_bank: usize,
        num_banks: usize,
        allocatable_regs: RegBitSet,
    ) -> Self {
        assert!(regs_per_bank <= 64, "Too many registers per bank");
        assert!(num_banks <= MAX_REGISTER_BANKS, "Too many register banks");
        
        let total_regs = regs_per_bank * num_banks;
        
        Self {
            regs_per_bank,
            total_regs,
            allocatable: allocatable_regs,
            used: RegBitSet::new(),
            fixed: RegBitSet::new(),
            clobbered: RegBitSet::new(),
            clocks: [0; MAX_REGISTER_BANKS],
            assignments: vec![None; total_regs],
            lock_counts: vec![0; total_regs],
            spill_callback: None,
        }
    }

    /// Set the spill callback for handling register eviction.
    pub fn set_spill_callback<F>(&mut self, callback: F)
    where
        F: Fn(AsmReg, &Assignment) -> Result<(), String> + 'static,
    {
        self.spill_callback = Some(Box::new(callback));
    }

    /// Allocate a register from the specified bank.
    ///
    /// First tries to find a free register, then uses clock algorithm
    /// to find an evictable register if necessary.
    pub fn allocate_reg(
        &mut self,
        bank: RegBank,
        local_idx: ValLocalIdx,
        part: u32,
        exclude: Option<&RegBitSet>,
    ) -> Result<AsmReg, RegAllocError> {
        let default_exclude = RegBitSet::new();
        let exclude = exclude.unwrap_or(&default_exclude);

        // First try to find a completely free register
        if let Some(reg_id) = self.find_first_free_in_bank(bank, exclude) {
            let reg = AsmReg::new(bank, reg_id);
            self.assign_register(reg, local_idx, part)?;
            return Ok(reg);
        }

        // No free registers, use clock algorithm to find evictable register
        if let Some(reg) = self.find_clocked_evictable(bank, exclude) {
            self.evict_register(reg)?;
            self.assign_register(reg, local_idx, part)?;
            return Ok(reg);
        }

        Err(RegAllocError::NoRegistersAvailable)
    }

    /// Find first free register in bank, excluding specified registers.
    fn find_first_free_in_bank(&self, bank: RegBank, exclude: &RegBitSet) -> Option<RegId> {
        // Free registers are allocatable but not used
        let mut free_regs = self.allocatable.clone();
        free_regs.intersect(&RegBitSet::all_in_bank(bank, self.regs_per_bank as u8));
        
        // Remove used and excluded registers
        let mut unavailable = self.used.clone();
        unavailable.union(exclude);
        
        for i in 0..MAX_REGISTER_BANKS {
            free_regs.banks[i] &= !unavailable.banks[i];
        }

        free_regs.find_first_in_bank(bank, &RegBitSet::new())
    }

    /// Find evictable register using clock algorithm.
    fn find_clocked_evictable(&mut self, bank: RegBank, exclude: &RegBitSet) -> Option<AsmReg> {
        if bank as usize >= MAX_REGISTER_BANKS {
            return None;
        }

        let _start_clock = self.clocks[bank as usize];
        
        for _ in 0..self.regs_per_bank {
            let reg = AsmReg::new(bank, self.clocks[bank as usize]);
            
            // Advance clock for next allocation
            self.clocks[bank as usize] = (self.clocks[bank as usize] + 1) % (self.regs_per_bank as RegId);
            
            // Check if this register can be evicted
            if self.allocatable.contains(reg) &&
               self.used.contains(reg) &&
               !self.fixed.contains(reg) &&
               !exclude.contains(reg) {
                return Some(reg);
            }
        }

        None
    }

    /// Assign a register to a value.
    fn assign_register(&mut self, reg: AsmReg, local_idx: ValLocalIdx, part: u32) -> Result<(), RegAllocError> {
        let linear_idx = reg.linear_index(self.regs_per_bank);
        if linear_idx >= self.total_regs {
            return Err(RegAllocError::InvalidRegister);
        }

        self.used.set(reg);
        self.assignments[linear_idx] = Some(Assignment::new(local_idx, part));
        Ok(())
    }

    /// Evict a register, spilling its contents if necessary.
    fn evict_register(&mut self, reg: AsmReg) -> Result<(), RegAllocError> {
        let linear_idx = reg.linear_index(self.regs_per_bank);
        if linear_idx >= self.total_regs {
            return Err(RegAllocError::InvalidRegister);
        }

        if let Some(assignment) = self.assignments[linear_idx] {
            // Call spill callback if register is modified
            if self.clobbered.contains(reg) {
                if let Some(ref spill_fn) = self.spill_callback {
                    spill_fn(reg, &assignment).map_err(|_| RegAllocError::NoRegistersAvailable)?;
                }
            }
        }

        self.free_register(reg)
    }

    /// Free a register without spilling.
    pub fn free_register(&mut self, reg: AsmReg) -> Result<(), RegAllocError> {
        let linear_idx = reg.linear_index(self.regs_per_bank);
        if linear_idx >= self.total_regs {
            return Err(RegAllocError::InvalidRegister);
        }

        if !self.used.contains(reg) {
            return Err(RegAllocError::RegisterNotAllocated);
        }

        self.used.clear(reg);
        self.fixed.clear(reg);
        self.clobbered.clear(reg);
        self.assignments[linear_idx] = None;
        self.lock_counts[linear_idx] = 0;
        Ok(())
    }

    /// Lock a register to prevent eviction.
    pub fn lock_register(&mut self, reg: AsmReg) -> Result<(), RegAllocError> {
        let linear_idx = reg.linear_index(self.regs_per_bank);
        if linear_idx >= self.total_regs {
            return Err(RegAllocError::InvalidRegister);
        }

        if !self.used.contains(reg) {
            return Err(RegAllocError::RegisterNotAllocated);
        }

        self.lock_counts[linear_idx] += 1;
        self.fixed.set(reg);
        Ok(())
    }

    /// Unlock a register, allowing eviction when lock count reaches zero.
    pub fn unlock_register(&mut self, reg: AsmReg) -> Result<(), RegAllocError> {
        let linear_idx = reg.linear_index(self.regs_per_bank);
        if linear_idx >= self.total_regs {
            return Err(RegAllocError::InvalidRegister);
        }

        if self.lock_counts[linear_idx] == 0 {
            return Err(RegAllocError::LockCountUnderflow);
        }

        self.lock_counts[linear_idx] -= 1;
        if self.lock_counts[linear_idx] == 0 {
            self.fixed.clear(reg);
        }
        Ok(())
    }

    /// Mark a register as clobbered (modified).
    pub fn mark_clobbered(&mut self, reg: AsmReg) {
        if self.used.contains(reg) {
            self.clobbered.set(reg);
        }
    }

    /// Get the assignment for a register.
    pub fn get_assignment(&self, reg: AsmReg) -> Option<Assignment> {
        let linear_idx = reg.linear_index(self.regs_per_bank);
        if linear_idx < self.total_regs {
            self.assignments[linear_idx]
        } else {
            None
        }
    }

    /// Check if register is currently allocated.
    pub fn is_allocated(&self, reg: AsmReg) -> bool {
        self.used.contains(reg)
    }

    /// Check if register is locked.
    pub fn is_locked(&self, reg: AsmReg) -> bool {
        self.fixed.contains(reg)
    }

    /// Check if register is clobbered.
    pub fn is_clobbered(&self, reg: AsmReg) -> bool {
        self.clobbered.contains(reg)
    }

    /// Get register usage statistics for a bank.
    pub fn bank_usage(&self, bank: RegBank) -> (u32, u32, u32) {
        let total = self.allocatable.count_in_bank(bank);
        let used = self.used.count_in_bank(bank);
        let fixed = self.fixed.count_in_bank(bank);
        (used, fixed, total)
    }

    /// Reset register file for new function.
    pub fn reset(&mut self) {
        self.used.clear_all();
        self.fixed.clear_all();
        self.clobbered.clear_all();
        self.clocks.fill(0);
        self.assignments.fill(None);
        self.lock_counts.fill(0);
    }

    /// Find a register currently assigned to the given value/part.
    pub fn find_register_for_value(&self, local_idx: ValLocalIdx, part: u32) -> Option<AsmReg> {
        for (i, assignment) in self.assignments.iter().enumerate() {
            if let Some(Assignment { local_idx: idx, part: p }) = assignment {
                if *idx == local_idx && *p == part {
                    return Some(AsmReg::from_linear_index(i, self.regs_per_bank));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_regfile() -> RegisterFile {
        // Create register file with 8 GP regs (bank 0) and 8 FP regs (bank 1)
        let mut allocatable = RegBitSet::new();
        allocatable.union(&RegBitSet::all_in_bank(0, 8)); // GP regs 0-7
        allocatable.union(&RegBitSet::all_in_bank(1, 8)); // FP regs 0-7
        
        RegisterFile::new(8, 2, allocatable)
    }

    #[test]
    fn test_regbitset_operations() {
        let mut set = RegBitSet::new();
        let reg = AsmReg::new(0, 5);
        
        assert!(!set.contains(reg));
        set.set(reg);
        assert!(set.contains(reg));
        set.clear(reg);
        assert!(!set.contains(reg));
    }

    #[test]
    fn test_register_allocation() {
        let mut regfile = create_test_regfile();
        
        // Allocate first register in GP bank
        let reg1 = regfile.allocate_reg(0, 100, 0, None).unwrap();
        assert_eq!(reg1.bank, 0);
        assert!(regfile.is_allocated(reg1));
        
        // Allocate second register
        let reg2 = regfile.allocate_reg(0, 101, 0, None).unwrap();
        assert_eq!(reg2.bank, 0);
        assert_ne!(reg1.id, reg2.id);
        
        // Check assignments
        assert_eq!(regfile.get_assignment(reg1).unwrap().local_idx, 100);
        assert_eq!(regfile.get_assignment(reg2).unwrap().local_idx, 101);
    }

    #[test]
    fn test_register_locking() {
        let mut regfile = create_test_regfile();
        let reg = regfile.allocate_reg(0, 100, 0, None).unwrap();
        
        assert!(!regfile.is_locked(reg));
        regfile.lock_register(reg).unwrap();
        assert!(regfile.is_locked(reg));
        
        regfile.unlock_register(reg).unwrap();
        assert!(!regfile.is_locked(reg));
    }

    #[test]
    fn test_register_eviction() {
        let mut regfile = create_test_regfile();
        
        // Fill all GP registers
        let mut regs = Vec::new();
        for i in 0..8 {
            let reg = regfile.allocate_reg(0, i as ValLocalIdx, 0, None).unwrap();
            regs.push(reg);
        }
        
        // All 8 registers should be allocated
        let (used, _, total) = regfile.bank_usage(0);
        assert_eq!(used, 8);
        assert_eq!(total, 8);
        
        // Next allocation should trigger eviction
        let new_reg = regfile.allocate_reg(0, 100, 0, None).unwrap();
        assert!(regfile.is_allocated(new_reg));
        
        // Should still have exactly 8 registers allocated total
        let (used_after, _, _) = regfile.bank_usage(0);
        assert_eq!(used_after, 8);
        
        // The new register should be one that was previously allocated to another value
        assert!(regs.contains(&new_reg));
    }

    #[test]
    fn test_find_register_for_value() {
        let mut regfile = create_test_regfile();
        let reg = regfile.allocate_reg(0, 100, 1, None).unwrap();
        
        assert_eq!(regfile.find_register_for_value(100, 1), Some(reg));
        assert_eq!(regfile.find_register_for_value(100, 0), None);
        assert_eq!(regfile.find_register_for_value(99, 1), None);
    }

    #[test]
    fn test_bank_usage_stats() {
        let mut regfile = create_test_regfile();
        
        let reg1 = regfile.allocate_reg(0, 100, 0, None).unwrap();
        let reg2 = regfile.allocate_reg(0, 101, 0, None).unwrap();
        regfile.lock_register(reg1).unwrap();
        
        let (used, fixed, total) = regfile.bank_usage(0);
        assert_eq!(used, 2);  // Two registers allocated
        assert_eq!(fixed, 1); // One register locked
        assert_eq!(total, 8); // Eight total registers available
    }
}