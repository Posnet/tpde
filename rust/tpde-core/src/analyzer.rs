use crate::adaptor::IrAdaptor;
use core::marker::PhantomData;

/// Computes block layout and liveness information for a function.
///
/// The analyzer walks the IR provided by [`IrAdaptor`] in a reverse
/// post-order and records begin/end positions for each value.  This data drives
/// the register allocator within [`CompilerBase`].  The algorithm closely
/// follows the description in the C++ docs and is summarized in
/// [`overview`].  The current implementation only provides the outer
/// skeleton.
#[allow(dead_code)]
pub struct Analyzer<A: IrAdaptor> {
    _marker: PhantomData<A>,
}

impl<A: IrAdaptor> Analyzer<A> {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    /// Build block layout and liveness for the given function using the adaptor.
    pub fn switch_func(&mut self, adaptor: &mut A, func: A::FuncRef) {
        let _ = adaptor.switch_func(func);
        // detailed analysis to be implemented later
        todo!()
    }
}
