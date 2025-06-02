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
//! More features like operand access will be added as the runtime expands.
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

    /// Iterator over all blocks in the current function.
    fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_>;

    /// Successor blocks of a given block.
    fn block_succs(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_>;

    /// Iterator over instructions contained in a block.
    fn block_insts(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_>;

    /// Result values produced by an instruction.
    fn inst_results(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_>;

    /// Operands referenced by an instruction.
    fn inst_operands(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_>;

    /// Local index of a value for liveness tracking.
    fn val_local_idx(&self, val: Self::ValueRef) -> usize;

    /// Should this value be ignored during liveness analysis?
    fn val_ignore_liveness(&self, val: Self::ValueRef) -> bool;
}
