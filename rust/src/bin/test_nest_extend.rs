use bumpalo::Bump;
use tpde::core::{session::CompilationSession, Analyzer, IrAdaptor};
use tpde::test_ir::{TestIR, TestIRAdaptor};

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

    // Create arena and session for analyzer
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new(&session);

    // Process the function
    let func = adaptor.funcs().next().unwrap();
    adaptor.switch_func(func);
    analyzer.switch_func(&mut adaptor, func);

    println!("RPO order for nest_extend:");
    for (idx, &block) in analyzer.order().iter().enumerate() {
        println!("  {}: {}", idx, adaptor.block_name(block));
    }

    println!("\nLiveness for %a:");
    let a_liveness = analyzer.liveness(0).unwrap();
    println!("  first: {}, last: {}", a_liveness.first, a_liveness.last);

    println!("\nExpected: 0->3 (entry->loop_inner2)");
    println!("Actual: {}->{}", a_liveness.first, a_liveness.last);
}
