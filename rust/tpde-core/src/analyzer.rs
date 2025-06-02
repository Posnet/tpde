use crate::adaptor::IrAdaptor;
use core::marker::PhantomData;

/// Computes block layout and liveness information for a function.
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
