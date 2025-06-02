/// Trait describing the interface between TPDE and a user IR.
///
/// Types implementing this trait provide references to functions,
/// blocks and values and allow the compiler to query basic
/// information about them. Only a small subset of the full C++
/// interface is modeled here so far.
pub trait IrAdaptor {
    type ValueRef: Copy + Eq;
    type InstRef: Copy + Eq;
    type BlockRef: Copy + Eq;
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
}
