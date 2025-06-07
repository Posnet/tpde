//! TestIR adaptor implementation for the TPDE compiler.
//!
//! This adaptor allows the TPDE compiler to work with TestIR,
//! enabling testing of compiler components with simple test cases.

use super::{Operation, TestIR, ValueType};
use crate::core::IrAdaptor;

/// Type aliases for TestIR references
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncRef(pub u32);

/// Adaptor that implements IrAdaptor for TestIR
pub struct TestIRAdaptor<'ir> {
    ir: &'ir TestIR,
    cur_func: u32,
}

impl<'ir> TestIRAdaptor<'ir> {
    pub fn new(ir: &'ir TestIR) -> Self {
        Self { ir, cur_func: 0 }
    }

    /// Get the current function index
    pub fn cur_func(&self) -> u32 {
        self.cur_func
    }

    /// Check if a value is a PHI node
    pub fn val_is_phi(&self, val: ValueRef) -> bool {
        self.ir.values[val.0 as usize].value_type == ValueType::Phi
    }

    /// Get the name of a value
    pub fn value_name(&self, val: ValueRef) -> &str {
        &self.ir.values[val.0 as usize].name
    }

    /// Get the name of a block
    pub fn block_name(&self, block: BlockRef) -> &str {
        &self.ir.blocks[block.0 as usize].name
    }

    /// Check if a function is extern
    pub fn func_extern(&self, func: FuncRef) -> bool {
        self.ir.functions[func.0 as usize].declaration
    }

    /// Check if a function is local only
    pub fn func_only_local(&self, func: FuncRef) -> bool {
        self.ir.functions[func.0 as usize].local_only
    }

    /// Get current function's arguments
    pub fn cur_args(&self) -> impl Iterator<Item = ValueRef> {
        let func = &self.ir.functions[self.cur_func as usize];
        (func.arg_begin_idx..func.arg_end_idx).map(ValueRef)
    }

    /// Get static allocas in the entry block
    pub fn cur_static_allocas(&self) -> Vec<ValueRef> {
        let func = &self.ir.functions[self.cur_func as usize];
        if func.block_begin_idx == func.block_end_idx {
            return Vec::new();
        }

        let block = &self.ir.blocks[func.block_begin_idx as usize];
        (block.inst_begin_idx..block.inst_end_idx)
            .filter(|&idx| self.ir.values[idx as usize].op == Operation::Alloca)
            .map(ValueRef)
            .collect()
    }

    /// Get alloca size
    pub fn val_alloca_size(&self, value: ValueRef) -> u32 {
        let val_idx = value.0 as usize;
        assert_eq!(self.ir.values[val_idx].op, Operation::Alloca);
        let op_idx = self.ir.values[val_idx].op_begin_idx as usize;
        self.ir.value_operands[op_idx]
    }

    /// Get alloca alignment
    pub fn val_alloca_align(&self, value: ValueRef) -> u32 {
        let val_idx = value.0 as usize;
        assert_eq!(self.ir.values[val_idx].op, Operation::Alloca);
        let op_idx = (self.ir.values[val_idx].op_begin_idx + 1) as usize;
        self.ir.value_operands[op_idx]
    }

    /// Get PHI nodes in a block
    pub fn block_phis(&self, block: BlockRef) -> impl Iterator<Item = ValueRef> {
        let block_info = &self.ir.blocks[block.0 as usize];
        (block_info.inst_begin_idx..block_info.phi_end_idx).map(ValueRef)
    }

    /// Get PHI info for a value
    pub fn val_as_phi(&self, value: ValueRef) -> TestPhiInfo {
        let val_idx = value.0 as usize;
        assert_eq!(self.ir.values[val_idx].value_type, ValueType::Phi);
        let info = &self.ir.values[val_idx];

        TestPhiInfo {
            ir: self.ir,
            op_begin_idx: info.op_begin_idx,
            op_count: info.op_count,
        }
    }

    /// Get all functions in the module
    pub fn funcs(&self) -> impl Iterator<Item = FuncRef> {
        (0..self.ir.functions.len()).map(|i| FuncRef(i as u32))
    }

    /// Get all blocks in the current function
    pub fn cur_blocks(&self) -> impl Iterator<Item = BlockRef> {
        let func = &self.ir.functions[self.cur_func as usize];
        (func.block_begin_idx..func.block_end_idx).map(BlockRef)
    }

    /// Get the entry block of the current function
    pub fn cur_entry_block(&self) -> BlockRef {
        let func = &self.ir.functions[self.cur_func as usize];
        assert!(func.block_begin_idx != func.block_end_idx);
        BlockRef(func.block_begin_idx)
    }
}

impl<'ir> IrAdaptor for TestIRAdaptor<'ir> {
    type ValueRef = ValueRef;
    type InstRef = InstRef;
    type BlockRef = BlockRef;
    type FuncRef = FuncRef;

    const INVALID_VALUE_REF: Self::ValueRef = ValueRef(!0);
    const INVALID_BLOCK_REF: Self::BlockRef = BlockRef(!0);
    const INVALID_FUNC_REF: Self::FuncRef = FuncRef(!0);

    // Tell the analyzer to visit function arguments during liveness analysis
    const TPDE_LIVENESS_VISIT_ARGS: bool = true;

    fn func_count(&self) -> u32 {
        self.ir.functions.len() as u32
    }

    fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_> {
        Box::new((0..self.ir.functions.len()).map(|i| FuncRef(i as u32)))
    }

    fn func_link_name(&self, func: Self::FuncRef) -> &str {
        &self.ir.functions[func.0 as usize].name
    }

    fn switch_func(&mut self, func: Self::FuncRef) -> bool {
        self.cur_func = func.0;
        true
    }

    fn reset(&mut self) {
        // Nothing to reset for TestIR
    }

    fn entry_block(&self) -> Self::BlockRef {
        let func = &self.ir.functions[self.cur_func as usize];
        assert!(func.block_begin_idx != func.block_end_idx);
        BlockRef(func.block_begin_idx)
    }

    fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
        let func = &self.ir.functions[self.cur_func as usize];
        Box::new((func.block_begin_idx..func.block_end_idx).map(BlockRef))
    }

    fn block_insts(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_> {
        let block_info = &self.ir.blocks[block.0 as usize];
        Box::new((block_info.phi_end_idx..block_info.inst_end_idx).map(InstRef))
    }

    fn block_succs(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
        let block_info = &self.ir.blocks[block.0 as usize];
        Box::new(
            (block_info.succ_begin_idx..block_info.succ_end_idx)
                .map(move |idx| BlockRef(self.ir.value_operands[idx as usize])),
        )
    }

    fn inst_operands(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        let info = &self.ir.values[inst.0 as usize];
        Box::new(
            (0..info.op_count)
                .map(move |i| ValueRef(self.ir.value_operands[(info.op_begin_idx + i) as usize])),
        )
    }

    fn inst_results(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        let info = &self.ir.values[inst.0 as usize];
        let is_def = info.op.info().is_def;
        if is_def {
            Box::new(std::iter::once(ValueRef(inst.0)))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn val_local_idx(&self, val: Self::ValueRef) -> usize {
        let func = &self.ir.functions[self.cur_func as usize];
        // Handle values that might be defined before the function (like constants)
        if val.0 < func.arg_begin_idx {
            // Map global constants to indices starting after all possible local values
            // This is a bit hacky but matches what the C++ code seems to do
            val.0 as usize
        } else {
            (val.0 - func.arg_begin_idx) as usize
        }
    }

    fn val_ignore_liveness(&self, val: Self::ValueRef) -> bool {
        self.ir.values[val.0 as usize].op == Operation::Alloca
    }

    fn cur_args(&self) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        let func = &self.ir.functions[self.cur_func as usize];
        Box::new((func.arg_begin_idx..func.arg_end_idx).map(ValueRef))
    }

    fn block_phis(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        let block_info = &self.ir.blocks[block.0 as usize];
        Box::new((block_info.inst_begin_idx..block_info.phi_end_idx).map(ValueRef))
    }

    fn val_is_phi(&self, val: Self::ValueRef) -> bool {
        self.ir.values[val.0 as usize].value_type == ValueType::Phi
    }

    fn phi_incoming_count(&self, phi: Self::ValueRef) -> u32 {
        let val_idx = phi.0 as usize;
        assert_eq!(self.ir.values[val_idx].value_type, ValueType::Phi);
        self.ir.values[val_idx].op_count
    }

    fn phi_incoming_val_for_slot(&self, phi: Self::ValueRef, slot: u32) -> Self::ValueRef {
        let val_idx = phi.0 as usize;
        assert_eq!(self.ir.values[val_idx].value_type, ValueType::Phi);
        let info = &self.ir.values[val_idx];
        assert!(slot < info.op_count);
        ValueRef(self.ir.value_operands[(info.op_begin_idx + slot) as usize])
    }

    fn phi_incoming_block_for_slot(&self, phi: Self::ValueRef, slot: u32) -> Self::BlockRef {
        let val_idx = phi.0 as usize;
        assert_eq!(self.ir.values[val_idx].value_type, ValueType::Phi);
        let info = &self.ir.values[val_idx];
        assert!(slot < info.op_count);
        BlockRef(self.ir.value_operands[(info.op_begin_idx + info.op_count + slot) as usize])
    }

    fn block_name(&self, block: Self::BlockRef) -> &str {
        &self.ir.blocks[block.0 as usize].name
    }

    fn set_block_idx(&self, _block: Self::BlockRef, _idx: usize) {
        // For TestIR, we don't need to store this information as we use block indices directly
    }
}

/// PHI node information for TestIR
pub struct TestPhiInfo<'ir> {
    ir: &'ir TestIR,
    op_begin_idx: u32,
    op_count: u32,
}

impl<'ir> TestPhiInfo<'ir> {
    pub fn incoming_count(&self) -> u32 {
        self.op_count
    }

    pub fn incoming_val_for_slot(&self, slot: u32) -> ValueRef {
        assert!(slot < self.op_count);
        ValueRef(self.ir.value_operands[(self.op_begin_idx + slot) as usize])
    }

    pub fn incoming_block_for_slot(&self, slot: u32) -> BlockRef {
        assert!(slot < self.op_count);
        BlockRef(self.ir.value_operands[(self.op_begin_idx + self.op_count + slot) as usize])
    }

    pub fn incoming_val_for_block(&self, block: BlockRef) -> ValueRef {
        for i in 0..self.op_count {
            if self.ir.value_operands[(self.op_begin_idx + self.op_count + i) as usize] == block.0 {
                return ValueRef(self.ir.value_operands[(self.op_begin_idx + i) as usize]);
            }
        }
        ValueRef(!0) // INVALID_VALUE_REF
    }
}
