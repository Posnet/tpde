// This module defines the IrAdaptor trait, which serves as the bridge between TPDE and
// any SSA-based intermediate representation (IR). The trait provides a minimal interface
// for the compiler to query IR structure: functions, blocks, instructions, and their
// relationships. It defines associated types for value/instruction/block/function references,
// methods to enumerate and switch between functions, iterate over blocks and instructions,
// query operands and results, and track value liveness. The design assumes SSA form with
// single-entry functions, basic blocks ending in terminators, multi-result instructions,
// and phi nodes for control flow. This abstraction allows TPDE to work with different
// IRs (like LLVM) without depending on their specific implementation details.

//! IRAdaptor responsibilities.
//!
//! The adaptor is the glue between TPDE and the user's SSA based IR. It exposes
//! the IR structure through a trait in Rust rather than the C++ concept described
//! in the original docs. The framework assumes:
//! - Each function has a single entry block.
//! - Basic blocks contain a list of instructions ending with a terminator.
//! - Instructions can produce multiple result values.
//! - Control flow edges use phi nodes; direct block arguments are planned.
//!
//! The `IrAdaptor` trait covers the minimal set of queries for now:
//! - Associated reference types for values, instructions, blocks and functions.
//! - Constants representing invalid references.
//! - Queries for function count and iteration.
//! - Methods to fetch linkage names, switch functions and reset between runs.
//!
//! Methods for operand access and block iteration are expected to grow over time
//! as the runtime expands.
//! Implementations may preprocess data in `switch_func` to speed up later calls.

/// Bridge between an SSA IR and TPDE.
///
/// The [`IrAdaptor`] trait provides the hooks the compiler needs to access an
/// arbitrary SSA IR.  In the C++ implementation this is expressed using
/// concepts; here we capture just the core subset as a trait.  The adaptor is
/// responsible for enumerating functions, blocks and instructions and for
/// allowing the compiler to obtain linkage names and switch the currently
/// compiled function.  More background can be found in [`guide`].
pub trait IrAdaptor {
    type ValueRef: Copy + Eq;
    type InstRef: Copy + Eq;
    type BlockRef: Copy + Eq + core::hash::Hash;
    type FuncRef: Copy + Eq;

    const INVALID_VALUE_REF: Self::ValueRef;
    const INVALID_BLOCK_REF: Self::BlockRef;
    const INVALID_FUNC_REF: Self::FuncRef;
    
    /// Whether the analyzer should visit function arguments during liveness analysis.
    const TPDE_LIVENESS_VISIT_ARGS: bool = false;

    /// Number of functions contained in the module.
    fn func_count(&self) -> u32;

    /// Iterator over all functions in the module.
    fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_>;

    /// Linkage name of the function.
    fn func_link_name(&self, func: Self::FuncRef) -> &str;

    /// Switch to the given function before compilation.
    fn switch_func(&mut self, func: Self::FuncRef) -> bool;

    /// Reset internal state between compilation runs.
    fn reset(&mut self);

    /// Entry block of the currently selected function.
    fn entry_block(&self) -> Self::BlockRef;

    /// Iterator over blocks in the current function.
    ///
    /// ```ignore
    /// # let mut adaptor = unimplemented!(); // see docs/tpde/guide.md for example
    /// adaptor.switch_func(adaptor.funcs().next().unwrap());
    /// for block in adaptor.blocks() {
    ///     for inst in adaptor.block_insts(block) {
    ///         let _ = adaptor.inst_operands(inst).count();
    ///     }
    /// }
    /// ```
    fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_>;

    /// Iterator over instructions of the given block.
    fn block_insts(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_>;

    /// Successor blocks of a given block.
    fn block_succs(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_>;

    /// Iterator over the operands of an instruction.
    fn inst_operands(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_>;

    /// Iterator over the result values produced by an instruction.
    fn inst_results(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_>;

    /// Local index of a value for liveness tracking.
    fn val_local_idx(&self, val: Self::ValueRef) -> usize;

    /// Should this value be ignored during liveness analysis?
    fn val_ignore_liveness(&self, val: Self::ValueRef) -> bool;
    
    /// Get the arguments of the current function.
    fn cur_args(&self) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        Box::new(std::iter::empty())
    }
    
    /// Get PHI nodes in a block.
    fn block_phis(&self, _block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
        Box::new(std::iter::empty())
    }
    
    /// Check if a value is a PHI node.
    fn val_is_phi(&self, _val: Self::ValueRef) -> bool {
        false
    }
    
    /// Get PHI node incoming count.
    fn phi_incoming_count(&self, _phi: Self::ValueRef) -> u32 {
        0
    }
    
    /// Get incoming value for a PHI node at given slot.
    fn phi_incoming_val_for_slot(&self, _phi: Self::ValueRef, _slot: u32) -> Self::ValueRef {
        Self::INVALID_VALUE_REF
    }
    
    /// Get incoming block for a PHI node at given slot.
    fn phi_incoming_block_for_slot(&self, _phi: Self::ValueRef, _slot: u32) -> Self::BlockRef {
        Self::INVALID_BLOCK_REF
    }
    
    /// Get the name of a block (for printing).
    fn block_name(&self, _block: Self::BlockRef) -> &str {
        ""
    }
    
    /// Set the block index for a block (used by analyzer).
    fn set_block_idx(&self, block: Self::BlockRef, idx: usize);
}
