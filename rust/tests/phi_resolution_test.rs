//! Test PHI node resolution in the LLVM compiler.
//!
//! This test verifies that PHI nodes are correctly resolved during compilation,
//! including handling of parallel moves and simple cycles.

use bumpalo::Bump;
use inkwell::context::Context;
use inkwell::IntPredicate;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a function with a simple loop containing PHI nodes.
///
/// ```llvm
/// define i32 @sum_to_n(i32 %n) {
/// entry:
///   br label %loop
///
/// loop:
///   %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
///   %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
///   %i_next = add i32 %i, 1
///   %sum_next = add i32 %sum, %i
///   %done = icmp sge i32 %i_next, %n
///   br i1 %done, label %exit, label %loop
///
/// exit:
///   ret i32 %sum_next
/// }
/// ```
fn create_loop_with_phi(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("phi_test");
    let i32_type = context.i32_type();

    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("sum_to_n", fn_type, None);

    // Create basic blocks
    let entry_block = context.append_basic_block(function, "entry");
    let loop_block = context.append_basic_block(function, "loop");
    let exit_block = context.append_basic_block(function, "exit");

    let builder = context.create_builder();

    // Entry block: jump to loop
    builder.position_at_end(entry_block);
    builder.build_unconditional_branch(loop_block).unwrap();

    // Loop block: PHI nodes and loop body
    builder.position_at_end(loop_block);

    // Create PHI nodes
    let i_phi = builder.build_phi(i32_type, "i").unwrap();
    let sum_phi = builder.build_phi(i32_type, "sum").unwrap();

    // Get the parameter
    let n_param = function.get_nth_param(0).unwrap().into_int_value();

    // Loop body
    let i_value = i_phi.as_basic_value().into_int_value();
    let sum_value = sum_phi.as_basic_value().into_int_value();

    let one = i32_type.const_int(1, false);
    let i_next = builder.build_int_add(i_value, one, "i_next").unwrap();
    let sum_next = builder
        .build_int_add(sum_value, i_value, "sum_next")
        .unwrap();

    // Loop condition
    let done = builder
        .build_int_compare(IntPredicate::SGE, i_next, n_param, "done")
        .unwrap();
    builder
        .build_conditional_branch(done, exit_block, loop_block)
        .unwrap();

    // Exit block
    builder.position_at_end(exit_block);
    builder.build_return(Some(&sum_next)).unwrap();

    // Add incoming values to PHI nodes
    let zero = i32_type.const_int(0, false);
    i_phi.add_incoming(&[(&zero, entry_block), (&i_next, loop_block)]);
    sum_phi.add_incoming(&[(&zero, entry_block), (&sum_next, loop_block)]);

    module
}

/// Create a function with PHI nodes that swap values.
/// This tests cycle detection and resolution.
///
/// ```llvm
/// define i32 @swap_test(i32 %a, i32 %b, i1 %cond) {
/// entry:
///   br i1 %cond, label %swap, label %no_swap
///
/// swap:
///   br label %merge
///
/// no_swap:
///   br label %merge
///
/// merge:
///   %x = phi i32 [ %b, %swap ], [ %a, %no_swap ]
///   %y = phi i32 [ %a, %swap ], [ %b, %no_swap ]
///   %result = add i32 %x, %y
///   ret i32 %result
/// }
/// ```
fn create_swap_phi_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("swap_test");
    let i32_type = context.i32_type();
    let i1_type = context.bool_type();

    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into(), i1_type.into()], false);
    let function = module.add_function("swap_test", fn_type, None);

    // Create basic blocks
    let entry_block = context.append_basic_block(function, "entry");
    let swap_block = context.append_basic_block(function, "swap");
    let no_swap_block = context.append_basic_block(function, "no_swap");
    let merge_block = context.append_basic_block(function, "merge");

    let builder = context.create_builder();

    // Get parameters
    let a_param = function.get_nth_param(0).unwrap().into_int_value();
    let b_param = function.get_nth_param(1).unwrap().into_int_value();
    let cond_param = function.get_nth_param(2).unwrap().into_int_value();

    // Entry block: conditional branch
    builder.position_at_end(entry_block);
    builder
        .build_conditional_branch(cond_param, swap_block, no_swap_block)
        .unwrap();

    // Swap block: just branch to merge
    builder.position_at_end(swap_block);
    builder.build_unconditional_branch(merge_block).unwrap();

    // No-swap block: just branch to merge
    builder.position_at_end(no_swap_block);
    builder.build_unconditional_branch(merge_block).unwrap();

    // Merge block: PHI nodes that potentially create a swap cycle
    builder.position_at_end(merge_block);

    let x_phi = builder.build_phi(i32_type, "x").unwrap();
    let y_phi = builder.build_phi(i32_type, "y").unwrap();

    // Add incoming values - this creates a swap pattern
    x_phi.add_incoming(&[(&b_param, swap_block), (&a_param, no_swap_block)]);
    y_phi.add_incoming(&[(&a_param, swap_block), (&b_param, no_swap_block)]);

    // Use both PHI results
    let x_value = x_phi.as_basic_value().into_int_value();
    let y_value = y_phi.as_basic_value().into_int_value();
    let result = builder.build_int_add(x_value, y_value, "result").unwrap();

    builder.build_return(Some(&result)).unwrap();

    module
}

#[test]
fn test_simple_loop_phi_resolution() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_loop_with_phi(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("sum_to_n").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("sum_to_n"));

    // Check that PHI nodes were resolved
    let stats = compiler.session().stats();
    assert!(
        stats.phi_nodes_resolved > 0,
        "Expected PHI nodes to be resolved"
    );

    println!("✅ Loop PHI resolution test passed!");
    println!("   Resolved {} PHI nodes", stats.phi_nodes_resolved);
    println!(
        "   Generated {} bytes of code",
        compiled_functions["sum_to_n"].code_size
    );
}

#[test]
fn test_phi_swap_cycle() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_swap_phi_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("swap_test").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("swap_test"));

    // Check that PHI nodes were resolved
    let stats = compiler.session().stats();
    assert!(
        stats.phi_nodes_resolved > 0,
        "Expected PHI nodes to be resolved"
    );

    println!("✅ PHI swap cycle test passed!");
    println!("   Resolved {} PHI nodes", stats.phi_nodes_resolved);
}

#[test]
fn test_multiple_predecessors_phi() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("multi_pred_test");
    let i32_type = context.i32_type();

    // Create a diamond-shaped CFG with PHI at the merge point
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("diamond", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    let left = context.append_basic_block(function, "left");
    let right = context.append_basic_block(function, "right");
    let merge = context.append_basic_block(function, "merge");

    let builder = context.create_builder();
    let n = function.get_nth_param(0).unwrap().into_int_value();

    // Entry: check if n > 5
    builder.position_at_end(entry);
    let five = i32_type.const_int(5, false);
    let cond = builder
        .build_int_compare(IntPredicate::SGT, n, five, "cond")
        .unwrap();
    builder.build_conditional_branch(cond, left, right).unwrap();

    // Left: multiply by 2
    builder.position_at_end(left);
    let two = i32_type.const_int(2, false);
    let left_val = builder.build_int_mul(n, two, "left_val").unwrap();
    builder.build_unconditional_branch(merge).unwrap();

    // Right: add 10
    builder.position_at_end(right);
    let ten = i32_type.const_int(10, false);
    let right_val = builder.build_int_add(n, ten, "right_val").unwrap();
    builder.build_unconditional_branch(merge).unwrap();

    // Merge: PHI node
    builder.position_at_end(merge);
    let phi = builder.build_phi(i32_type, "result").unwrap();
    phi.add_incoming(&[(&left_val, left), (&right_val, right)]);

    let result = phi.as_basic_value();
    builder.build_return(Some(&result)).unwrap();

    // Compile
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    compiler.compile_function_by_name("diamond").unwrap();

    let stats = compiler.session().stats();
    assert!(stats.phi_nodes_resolved > 0);

    println!("✅ Multiple predecessors PHI test passed!");
    println!("   Resolved {} PHI nodes", stats.phi_nodes_resolved);
}
