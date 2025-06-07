// This test suite validates PHI node compilation, one of the most complex aspects of SSA
// (Static Single Assignment) form compilation. PHI nodes appear at control flow merge points
// where different execution paths bring different values for the same variable. The tests
// cover three critical scenarios: 1) Simple PHI nodes in if/else patterns where the max
// function demonstrates basic value selection at merge points; 2) Loop PHI nodes in the
// sum_to_n function showing how loop-carried dependencies work with PHI nodes receiving
// values from both loop entry and back-edges; 3) PHI cycles where values are swapped
// between PHI nodes, requiring sophisticated resolution algorithms to avoid infinite loops
// during code generation. PHI node compilation is particularly challenging because it requires
// careful coordination of value movements, cycle detection to identify circular dependencies,
// and temporary register allocation to break cycles. These tests verify that TPDE correctly
// handles PHI nodes essential for compiling any non-trivial control flow. Note: Full PHI
// resolution with cycle breaking is not yet implemented in the Rust version, representing
// a critical gap for loop compilation.

//! Test PHI node compilation in LLVM compiler.

use bumpalo::Bump;
use inkwell::context::Context;
use inkwell::IntPredicate;
use tpde::core::CompilationSession;
use tpde::llvm::LlvmCompiler;

/// Create a test module with PHI nodes.
fn create_phi_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("phi_test");
    let i32_type = context.i32_type();

    // Create a function with conditional logic and PHI nodes
    // int max(int a, int b) {
    //   int result;
    //   if (a > b) {
    //     result = a;
    //   } else {
    //     result = b;
    //   }
    //   return result;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("max", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    let then_block = context.append_basic_block(function, "then");
    let else_block = context.append_basic_block(function, "else");
    let merge = context.append_basic_block(function, "merge");

    let builder = context.create_builder();

    // Entry block
    builder.position_at_end(entry);
    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let cond = builder
        .build_int_compare(IntPredicate::SGT, a, b, "cmp")
        .unwrap();
    builder
        .build_conditional_branch(cond, then_block, else_block)
        .unwrap();

    // Then block: result = a
    builder.position_at_end(then_block);
    builder.build_unconditional_branch(merge).unwrap();

    // Else block: result = b
    builder.position_at_end(else_block);
    builder.build_unconditional_branch(merge).unwrap();

    // Merge block with PHI node
    builder.position_at_end(merge);
    let phi = builder.build_phi(i32_type, "result").unwrap();
    phi.add_incoming(&[(&a, then_block), (&b, else_block)]);
    builder.build_return(Some(&phi.as_basic_value())).unwrap();

    module
}

/// Create a module with a loop containing PHI nodes.
fn create_loop_phi_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("loop_phi_test");
    let i32_type = context.i32_type();

    // Create a function that sums numbers from 1 to n
    // int sum(int n) {
    //   int i = 0;
    //   int sum = 0;
    //   while (i < n) {
    //     i = i + 1;
    //     sum = sum + i;
    //   }
    //   return sum;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("sum_to_n", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    let loop_header = context.append_basic_block(function, "loop.header");
    let loop_body = context.append_basic_block(function, "loop.body");
    let exit = context.append_basic_block(function, "exit");

    let builder = context.create_builder();

    // Entry block
    builder.position_at_end(entry);
    let n = function.get_nth_param(0).unwrap().into_int_value();
    let zero = i32_type.const_int(0, false);
    builder.build_unconditional_branch(loop_header).unwrap();

    // Loop header with PHI nodes
    builder.position_at_end(loop_header);
    let i_phi = builder.build_phi(i32_type, "i").unwrap();
    let sum_phi = builder.build_phi(i32_type, "sum").unwrap();

    // Add incoming values from entry
    i_phi.add_incoming(&[(&zero, entry)]);
    sum_phi.add_incoming(&[(&zero, entry)]);

    // Loop condition
    let i_val = i_phi.as_basic_value().into_int_value();
    let cond = builder
        .build_int_compare(IntPredicate::SLT, i_val, n, "cmp")
        .unwrap();
    builder
        .build_conditional_branch(cond, loop_body, exit)
        .unwrap();

    // Loop body
    builder.position_at_end(loop_body);
    let one = i32_type.const_int(1, false);
    let i_next = builder.build_int_add(i_val, one, "i.next").unwrap();
    let sum_val = sum_phi.as_basic_value().into_int_value();
    let sum_next = builder.build_int_add(sum_val, i_next, "sum.next").unwrap();
    builder.build_unconditional_branch(loop_header).unwrap();

    // Add incoming values from loop body
    i_phi.add_incoming(&[(&i_next, loop_body)]);
    sum_phi.add_incoming(&[(&sum_next, loop_body)]);

    // Exit block
    builder.position_at_end(exit);
    let final_sum = sum_phi.as_basic_value();
    builder.build_return(Some(&final_sum)).unwrap();

    module
}

#[test]
fn test_simple_phi_compilation() {
    let context = Context::create();
    let module = create_phi_test_module(&context);
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the max function
    compiler
        .compile_function_by_name("max")
        .expect("PHI compilation should succeed");

    // Verify compilation
    let compiled = compiler.compiled_functions().get("max").unwrap();
    assert_eq!(compiled.name, "max");
    assert!(compiled.code_size > 0, "Should generate code");

    // Check statistics
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("Phi"));
    assert!(stats.instruction_counts.contains_key("ICmp"));
    assert!(stats.instruction_counts.contains_key("Br"));

    println!("✅ Simple PHI test passed");
}

#[test]
fn test_loop_phi_compilation() {
    let context = Context::create();
    let module = create_loop_phi_module(&context);
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the sum function
    compiler
        .compile_function_by_name("sum_to_n")
        .expect("Loop PHI compilation should succeed");

    // Verify compilation
    let compiled = compiler.compiled_functions().get("sum_to_n").unwrap();
    assert_eq!(compiled.name, "sum_to_n");
    assert!(compiled.code_size > 0, "Should generate code");

    // Check statistics
    let stats = compiler.session().stats();
    assert_eq!(
        stats.instruction_counts["Phi"], 2,
        "Should have 2 PHI nodes"
    );
    assert!(
        stats.instruction_counts["Add"] >= 2,
        "Should have at least 2 add instructions"
    );

    println!("✅ Loop PHI test passed");
}

#[test]
fn test_phi_cycle_compilation() {
    let context = Context::create();
    let module = context.create_module("cycle_test");
    let i32_type = context.i32_type();

    // Create a function with PHI nodes that form a cycle
    // This tests the cycle detection in PHI resolution
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("phi_cycle", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    let loop_block = context.append_basic_block(function, "loop");
    let exit = context.append_basic_block(function, "exit");

    let builder = context.create_builder();

    // Entry
    builder.position_at_end(entry);
    let init1 = i32_type.const_int(1, false);
    let init2 = i32_type.const_int(2, false);
    builder.build_unconditional_branch(loop_block).unwrap();

    // Loop with swapping PHI nodes (creates cycle)
    builder.position_at_end(loop_block);
    let phi1 = builder.build_phi(i32_type, "val1").unwrap();
    let phi2 = builder.build_phi(i32_type, "val2").unwrap();

    phi1.add_incoming(&[(&init1, entry)]);
    phi2.add_incoming(&[(&init2, entry)]);

    // Create cycle: phi1 <- phi2, phi2 <- phi1
    let val1 = phi1.as_basic_value().into_int_value();
    let val2 = phi2.as_basic_value().into_int_value();

    // Some condition to exit
    let ten = i32_type.const_int(10, false);
    let sum = builder.build_int_add(val1, val2, "sum").unwrap();
    let cond = builder
        .build_int_compare(IntPredicate::SGT, sum, ten, "cond")
        .unwrap();
    builder
        .build_conditional_branch(cond, exit, loop_block)
        .unwrap();

    // Add cyclic incoming values
    phi1.add_incoming(&[(&val2, loop_block)]); // val1 <- val2
    phi2.add_incoming(&[(&val1, loop_block)]); // val2 <- val1

    // Exit
    builder.position_at_end(exit);
    builder.build_return(Some(&sum)).unwrap();

    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // This should compile without errors, handling the PHI cycle
    compiler
        .compile_function_by_name("phi_cycle")
        .expect("PHI cycle compilation should succeed");

    println!("✅ PHI cycle test passed");
}
