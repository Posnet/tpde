use tpde::adaptor::IrAdaptor;
use tpde::core::{Assembler, ElfAssembler};
use object::{File, Object};

struct DummyAdaptor;

impl IrAdaptor for DummyAdaptor {
    type ValueRef = ();
    type InstRef = ();
    type BlockRef = ();
    type FuncRef = ();

    const INVALID_VALUE_REF: Self::ValueRef = ();
    const INVALID_BLOCK_REF: Self::BlockRef = ();
    const INVALID_FUNC_REF: Self::FuncRef = ();

    fn func_count(&self) -> u32 { 0 }
    fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_> { Box::new(std::iter::empty()) }
    fn func_link_name(&self, _f: Self::FuncRef) -> &str { "" }
    fn switch_func(&mut self, _f: Self::FuncRef) -> bool { false }
    fn reset(&mut self) {}
    fn entry_block(&self) -> Self::BlockRef { () }
    fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> { Box::new(std::iter::empty()) }
    fn block_insts(&self, _: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_> { Box::new(std::iter::empty()) }
    fn block_succs(&self, _: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> { Box::new(std::iter::empty()) }
    fn inst_operands(&self, _: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> { Box::new(std::iter::empty()) }
    fn inst_results(&self, _: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> { Box::new(std::iter::empty()) }
    fn val_local_idx(&self, _: Self::ValueRef) -> usize { 0 }
    fn val_ignore_liveness(&self, _: Self::ValueRef) -> bool { false }
}

#[test]
fn simple_object() {
    let mut asm = <ElfAssembler as Assembler<DummyAdaptor>>::new(true);
    let lbl = <ElfAssembler as Assembler<DummyAdaptor>>::label_create(&mut asm);
    <ElfAssembler as Assembler<DummyAdaptor>>::label_place(&mut asm, lbl);
    asm.append(&[0xC3], 1); // ret
    let sym = <ElfAssembler as Assembler<DummyAdaptor>>::sym_predef_func(&mut asm, "foo", false, false);
    let off = asm.label_offset(lbl).unwrap();
    let sec = asm.current_section();
    asm.define_symbol(sym, sec, off, 1);
    let obj = asm.build_object();
    let file = File::parse(&*obj).unwrap();
    assert!(file.section_by_name(".text").is_some());
    assert!(file.symbol_by_name("foo").is_some());
}
