use tpde::test_ir::{TestIR, TestSpec};

fn main() {
    let content = std::fs::read_to_string("/Users/entropy/Developer/tpde/tpde/test/filetest/tir/simple.tir").unwrap();
    let spec = TestSpec::parse(&content).unwrap();
    let ir = TestIR::parse(&spec.tir_content).unwrap();
    println!("{}", ir.print());
}