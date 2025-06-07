use tpde::test_ir::{TestIR, TestIRAdaptor};
use tpde::core::{Analyzer, IrAdaptor};
use std::collections::HashMap;

fn main() {
    let tir_content = r#"
mixed_loop2() {
entry:
  jump ^loop1_head, ^cont1
loop1_head:
  jump ^loop1_body, ^cont1
cont1:
  jump ^loop2_head
loop1_body:
  jump ^loop1_head
loop2_body:
  jump ^loop2_head
loop2_head:
  jump ^loop2_body, ^ret
ret:
  terminate
}
"#;

    let ir = TestIR::parse(tir_content).expect("Failed to parse TIR");
    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new();
    
    // Process the function
    let func = adaptor.funcs().next().unwrap();
    adaptor.switch_func(func);
    analyzer.switch_func(&mut adaptor, func);
    
    println!("Block Layout for mixed_loop2");
    let layout = analyzer.block_layout();
    for (idx, &block) in layout.iter().enumerate() {
        let block_name = adaptor.block_name(block);
        println!("{}: {}", idx, block_name);
    }
    println!("End Block Layout");
    
    println!("Loops for mixed_loop2");
    let loops = analyzer.loops();
    for (idx, loop_info) in loops.iter().enumerate() {
        println!(
            "{}: level {}, parent {}, {}->{}",
            idx,
            loop_info.level,
            loop_info.parent,
            loop_info.begin,
            loop_info.end
        );
    }
    println!("End Loops");
    
    // Debug: Check edges
    println!("\nDebug: Block successors");
    for (idx, &block) in layout.iter().enumerate() {
        let block_name = adaptor.block_name(block);
        print!("{} ({}): ", idx, block_name);
        for succ in adaptor.block_succs(block) {
            let succ_name = adaptor.block_name(succ);
            print!("{} ", succ_name);
        }
        println!();
    }
}