use crate::adaptor::IrAdaptor;

/// Trait implemented by architecture specific assemblers.
pub trait Assembler<A: IrAdaptor> {
    type SymRef;
    type Label;

    /// Create a new assembler.
    fn new(generate_object: bool) -> Self
    where
        Self: Sized;

    fn label_create(&mut self) -> Self::Label;
    fn label_place(&mut self, label: Self::Label);

    fn sym_predef_func(&mut self, name: &str, local: bool, weak: bool) -> Self::SymRef;
    fn sym_add_undef(&mut self, name: &str, local: bool, weak: bool);
}
