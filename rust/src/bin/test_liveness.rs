use tpde::test_ir::{TestIR, TestIRAdaptor};
use tpde::core::{Analyzer, IrAdaptor};

fn main() {
    let tir_content = r#"
simple(%a) {
entry:
  %const =
  jump ^loop
loop:
  %b = phi [^entry, %const], [^loop, %e]
  %c = phi [^entry, %a], [^loop, %f]
  %d = %c
  %e = %b, %d
  %f = %c
  jump ^loop, ^ret
ret:
  %ret = %b
  terminate
}
"#;

    let ir = TestIR::parse(tir_content).expect("Failed to parse TIR");
    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new();
    
    // Process the function
    let func = adaptor.funcs().next().unwrap();
    adaptor.switch_func(func);
    
    println!("\nDebug: Processing function");
    
    // Manually trace through what the analyzer does
    println!("Arguments:");
    for arg in adaptor.cur_args() {
        println!("  - {}", adaptor.value_name(arg));
    }
    
    println!("\nBlocks in order:");
    adaptor.switch_func(func);
    analyzer.switch_func(&mut adaptor, func);
    for (idx, &block) in analyzer.order().iter().enumerate() {
        println!("  {}: {}", idx, adaptor.block_name(block));
    }
    
    // Print liveness info
    println!("Liveness for simple");
    
    // Debug: print all values to understand the indexing
    println!("\nDebug: All values in the IR:");
    for (idx, val) in ir.values.iter().enumerate() {
        println!("  {}: {} ({:?}, op={:?})", idx, val.name, val.value_type, val.op);
    }
    
    // Get block names for mapping
    let blocks: Vec<_> = analyzer.order().to_vec();
    let block_names: Vec<String> = blocks.iter()
        .map(|&b| adaptor.block_name(b).to_string())
        .collect();
    
    println!("\nLiveness analysis results:");
    
    let mut value_idx = 0;
    
    // Function arguments
    for arg in adaptor.cur_args() {
        let arg_name = adaptor.value_name(arg);
        if let Some(liveness) = analyzer.liveness(value_idx) {
            let first_block = block_names.get(liveness.first).map(|s| s.as_str()).unwrap_or("?");
            let last_block = block_names.get(liveness.last).map(|s| s.as_str()).unwrap_or("?");
            
            println!(
                "{}: {} refs, {}->{} ({}->{}), lf: {} [{}]",
                value_idx,
                liveness.ref_count,
                liveness.first,
                liveness.last,
                first_block,
                last_block,
                liveness.last_full,
                arg_name
            );
        } else {
            println!("{}: ignored [{}]", value_idx, arg_name);
        }
        value_idx += 1;
    }
    
    // Values in blocks (in RPO order)
    for &block in &blocks {
        let block_name = adaptor.block_name(block);
        println!("  Block: {}", block_name);
        
        // PHI nodes
        for phi in adaptor.block_phis(block) {
            let phi_name = adaptor.value_name(phi);
            if let Some(liveness) = analyzer.liveness(value_idx) {
                let first_block = block_names.get(liveness.first).map(|s| s.as_str()).unwrap_or("?");
                let last_block = block_names.get(liveness.last).map(|s| s.as_str()).unwrap_or("?");
                
                println!(
                    "  {}: {} refs, {}->{} ({}->{}), lf: {} [PHI {}]",
                    value_idx,
                    liveness.ref_count,
                    liveness.first,
                    liveness.last,
                    first_block,
                    last_block,
                    liveness.last_full,
                    phi_name
                );
            } else {
                println!("  {}: ignored [PHI {}]", value_idx, phi_name);
            }
            value_idx += 1;
        }
        
        // Regular instructions
        for inst in adaptor.block_insts(block) {
            let results: Vec<_> = adaptor.inst_results(inst).collect();
            if !results.is_empty() {
                let inst_name = adaptor.value_name(results[0]);
                if let Some(liveness) = analyzer.liveness(value_idx) {
                    let first_block = block_names.get(liveness.first).map(|s| s.as_str()).unwrap_or("?");
                    let last_block = block_names.get(liveness.last).map(|s| s.as_str()).unwrap_or("?");
                    
                    println!(
                        "  {}: {} refs, {}->{} ({}->{}), lf: {} [{}]",
                        value_idx,
                        liveness.ref_count,
                        liveness.first,
                        liveness.last,
                        first_block,
                        last_block,
                        liveness.last_full,
                        inst_name
                    );
                } else {
                    println!("  {}: ignored [{}]", value_idx, inst_name);
                }
            }
            value_idx += 1;
        }
    }
    
    println!("End Liveness");
}