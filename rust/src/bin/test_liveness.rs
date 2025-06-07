use tpde::test_ir::{TestIR, TestIRAdaptor};
use tpde::core::{Analyzer, IrAdaptor};

fn main() {
    let tir_content = r#"
nest_extend() {
entry:
  %a =
  jump ^loop1
loop1:
  jump ^loop_inner1, ^ret
loop_inner1:
  %b = %a
  jump ^loop_inner1, ^loop_inner2
loop_inner2:
  jump ^loop_inner2, ^loop1
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
    
    println!("Block Layout:");
    let layout = analyzer.block_layout();
    for (idx, &block) in layout.iter().enumerate() {
        let block_name = adaptor.block_name(block);
        println!("{idx}: {block_name}");
    }
    
    println!("\nLoops:");
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
    
    println!("\nLiveness for %a (local_idx 0):");
    if let Some(liveness) = analyzer.liveness(0) {
        println!("  first: {}, last: {}, last_full: {}", 
                 liveness.first, liveness.last, liveness.last_full);
        
        let first_block = layout.get(liveness.first)
            .map(|&b| adaptor.block_name(b))
            .unwrap_or("?");
        let last_block = layout.get(liveness.last)
            .map(|&b| adaptor.block_name(b))
            .unwrap_or("?");
            
        println!("  {first_block}->{last_block}");
    }
}