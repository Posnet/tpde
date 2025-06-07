use tpde::test_ir::{TestIR, TestIRAdaptor};
use tpde::core::{Analyzer, IrAdaptor};

fn main() {
    let test_ir = r#"
; Test the RPO fix
loop1() {
entry:
  jump ^loop_head
loop_head:
  jump ^loop_body, ^cont
loop_body:
  jump ^loop_head
cont:
  terminate
}
"#;

    // Parse the IR
    let ir = TestIR::parse(test_ir).expect("Failed to parse test IR");
    
    // Create adaptor and analyzer
    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new();
    
    // Process the function
    let funcs: Vec<_> = adaptor.funcs().collect();
    for func in funcs {
        println!("Processing function: {}", adaptor.func_link_name(func));
        adaptor.switch_func(func);
        analyzer.switch_func(&mut adaptor, func);
        
        // Print RPO
        println!("RPO order:");
        for (idx, &block) in analyzer.order().iter().enumerate() {
            let block_name = adaptor.block_name(block);
            println!("{}: {}", idx, block_name);
        }
        
        // Check if the order matches expected
        let rpo: Vec<&str> = analyzer.order().iter()
            .map(|&b| adaptor.block_name(b))
            .collect();
        
        let expected = vec!["entry", "loop_head", "loop_body", "cont"];
        
        if rpo == expected {
            println!("✓ RPO order is correct!");
        } else {
            println!("✗ RPO order is incorrect!");
            println!("  Expected: {:?}", expected);
            println!("  Got:      {:?}", rpo);
        }
    }
}