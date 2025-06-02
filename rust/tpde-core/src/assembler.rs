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

use object::write::{Object, SectionId, StandardSection, SymbolId, Symbol, SymbolSection, SymbolScope, SymbolKind};
use std::collections::HashMap;
use object::{BinaryFormat, Architecture, Endianness};

/// Label identifier used by [`ElfAssembler`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElfLabel(usize);

#[derive(Debug)]
struct LabelInfo {
    section: SectionId,
    offset: Option<u64>,
}

/// Minimal assembler based on the [`object`] crate producing ELF objects.
pub struct ElfAssembler {
    obj: Object<'static>,
    current: SectionId,
    labels: Vec<LabelInfo>,
    offsets: HashMap<SectionId, u64>,
    _emit_obj: bool,
}

impl ElfAssembler {
    /// Create a new section returning its identifier.
    pub fn add_section(&mut self, name: &str, kind: object::SectionKind) -> SectionId {
        let id = self.obj.add_section(Vec::new(), name.as_bytes().to_vec(), kind);
        self.offsets.insert(id, 0);
        id
    }

    /// Switch the current section.
    pub fn switch_section(&mut self, id: SectionId) {
        self.current = id;
    }

    /// Returns the current section identifier.
    pub fn current_section(&self) -> SectionId {
        self.current
    }

    /// Append data to the current section and return the offset.
    pub fn append(&mut self, data: &[u8], align: u64) -> u64 {
        let off = self.obj.append_section_data(self.current, data, align);
        let end = off + data.len() as u64;
        let entry = self.offsets.entry(self.current).or_insert(0);
        if *entry < end {
            *entry = end;
        }
        off
    }

    /// Finalize the object and return the binary.
    pub fn build_object(self) -> Vec<u8> {
        self.obj.write().expect("emit object")
    }

    /// Update a symbol to point at a location within a section.
    pub fn define_symbol(&mut self, sym: SymbolId, section: SectionId, offset: u64, size: u64) {
        let symbol = self.obj.symbol_mut(sym);
        symbol.section = SymbolSection::Section(section);
        symbol.value = offset;
        symbol.size = size;
    }

    fn current_offset(&self) -> u64 {
        *self.offsets.get(&self.current).unwrap_or(&0)
    }

    /// Get the offset of a label if placed.
    pub fn label_offset(&self, label: ElfLabel) -> Option<u64> {
        self.labels.get(label.0).and_then(|l| l.offset)
    }

    /// Return true if the label has not been placed yet.
    pub fn label_is_pending(&self, label: ElfLabel) -> bool {
        self.labels.get(label.0).map_or(false, |l| l.offset.is_none())
    }
}

impl<A: IrAdaptor> Assembler<A> for ElfAssembler {
    type SymRef = SymbolId;
    type Label = ElfLabel;

    fn new(generate_object: bool) -> Self {
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let text = obj.section_id(StandardSection::Text);
        let mut offsets = HashMap::new();
        offsets.insert(text, 0);
        Self {
            obj,
            current: text,
            labels: Vec::new(),
            offsets,
            _emit_obj: generate_object,
        }
    }

    fn label_create(&mut self) -> Self::Label {
        let id = self.labels.len();
        self.labels.push(LabelInfo { section: self.current, offset: None });
        ElfLabel(id)
    }

    fn label_place(&mut self, label: Self::Label) {
        let offset = self.current_offset();
        if let Some(info) = self.labels.get_mut(label.0) {
            info.section = self.current;
            info.offset = Some(offset);
        }
    }

    fn sym_predef_func(&mut self, name: &str, local: bool, weak: bool) -> Self::SymRef {
        let scope = if local { SymbolScope::Compilation } else { SymbolScope::Linkage };
        self.obj.add_symbol(Symbol {
            name: name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Text,
            scope,
            weak,
            section: SymbolSection::Undefined,
            flags: object::write::SymbolFlags::None,
        })
    }

    fn sym_add_undef(&mut self, name: &str, local: bool, weak: bool) {
        let scope = if local { SymbolScope::Compilation } else { SymbolScope::Linkage };
        self.obj.add_symbol(Symbol {
            name: name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Unknown,
            scope,
            weak,
            section: SymbolSection::Undefined,
            flags: object::write::SymbolFlags::None,
        });
    }

    fn finalize(&mut self) {}

    fn build_object_file(&mut self) -> Vec<u8> { self.obj.write().expect("emit object") }

    fn map<F>(&mut self, _resolve: F) -> bool
    where
        F: FnMut(&str) -> *const u8,
    {
        true
    }
}

