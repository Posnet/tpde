use crate::adaptor::IrAdaptor;

/// Trait implemented by architecture specific assemblers.
///
/// An assembler owns the sections, symbols and relocations for the generated
/// machine code.  At the end of compilation it can either emit an ELF object
/// or map the code directly into memory for JIT execution.  This is a thin
/// wrapper around the concept described in the original docs and summarised in
/// [`overview`].  Only a few entry points are defined for now.
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

    /// Finalize sections and relocations after code generation.
    fn finalize(&mut self);

    /// Write a finished object file to a byte vector.
    fn build_object_file(&mut self) -> Vec<u8>;

    /// Map the generated code into memory for JIT execution.
    ///
    /// `resolve` should return the address of any unresolved symbol.
    fn map<F>(&mut self, resolve: F) -> bool
    where
        F: FnMut(&str) -> *const u8;
}
