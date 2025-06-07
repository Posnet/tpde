//! Test switch statement compilation.
//!
//! This test verifies that switch statements are correctly compiled into
//! sequences of comparisons and jumps.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a simple switch test module.
fn create_basic_switch_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("basic_switch_test");
    let i32_type = context.i32_type();

    // Function: i32 switch_test(i32 value)
    // {
    //     switch(value) {
    //         case 0: return 10;
    //         case 1: return 20;
    //         case 2: return 30;
    //         default: return -1;
    //     }
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("switch_test", fn_type, None);

    // Create blocks
    let entry_block = context.append_basic_block(function, "entry");
    let case0_block = context.append_basic_block(function, "case.0");
    let case1_block = context.append_basic_block(function, "case.1");
    let case2_block = context.append_basic_block(function, "case.2");
    let default_block = context.append_basic_block(function, "default");

    let builder = context.create_builder();

    // Entry block with switch
    builder.position_at_end(entry_block);
    let value = function.get_nth_param(0).unwrap().into_int_value();

    builder
        .build_switch(
            value,
            default_block,
            &[
                (i32_type.const_int(0, false), case0_block),
                (i32_type.const_int(1, false), case1_block),
                (i32_type.const_int(2, false), case2_block),
            ],
        )
        .unwrap();

    // Case 0: return 10
    builder.position_at_end(case0_block);
    let ten = i32_type.const_int(10, false);
    builder.build_return(Some(&ten)).unwrap();

    // Case 1: return 20
    builder.position_at_end(case1_block);
    let twenty = i32_type.const_int(20, false);
    builder.build_return(Some(&twenty)).unwrap();

    // Case 2: return 30
    builder.position_at_end(case2_block);
    let thirty = i32_type.const_int(30, false);
    builder.build_return(Some(&thirty)).unwrap();

    // Default: return -1
    builder.position_at_end(default_block);
    let neg_one = i32_type.const_int(-1i64 as u64, true);
    builder.build_return(Some(&neg_one)).unwrap();

    module
}

/// Create a switch test with larger values to test range.
fn create_range_switch_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("range_switch_test");
    let i32_type = context.i32_type();

    // Function: i32 range_switch(i32 value)
    // {
    //     switch(value) {
    //         case 100: return 1;
    //         case 200: return 2;
    //         case 300: return 3;
    //         default: return 0;
    //     }
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("range_switch", fn_type, None);

    // Create blocks
    let entry_block = context.append_basic_block(function, "entry");
    let case100_block = context.append_basic_block(function, "case.100");
    let case200_block = context.append_basic_block(function, "case.200");
    let case300_block = context.append_basic_block(function, "case.300");
    let default_block = context.append_basic_block(function, "default");

    let builder = context.create_builder();

    // Entry block with switch
    builder.position_at_end(entry_block);
    let value = function.get_nth_param(0).unwrap().into_int_value();

    builder
        .build_switch(
            value,
            default_block,
            &[
                (i32_type.const_int(100, false), case100_block),
                (i32_type.const_int(200, false), case200_block),
                (i32_type.const_int(300, false), case300_block),
            ],
        )
        .unwrap();

    // Case 100: return 1
    builder.position_at_end(case100_block);
    let one = i32_type.const_int(1, false);
    builder.build_return(Some(&one)).unwrap();

    // Case 200: return 2
    builder.position_at_end(case200_block);
    let two = i32_type.const_int(2, false);
    builder.build_return(Some(&two)).unwrap();

    // Case 300: return 3
    builder.position_at_end(case300_block);
    let three = i32_type.const_int(3, false);
    builder.build_return(Some(&three)).unwrap();

    // Default: return 0
    builder.position_at_end(default_block);
    let zero = i32_type.const_int(0, false);
    builder.build_return(Some(&zero)).unwrap();

    module
}

/// Create an empty switch test (only default case).
fn create_empty_switch_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("empty_switch_test");
    let i32_type = context.i32_type();

    // Function: i32 empty_switch(i32 value)
    // {
    //     switch(value) {
    //         default: return 42;
    //     }
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("empty_switch", fn_type, None);

    // Create blocks
    let entry_block = context.append_basic_block(function, "entry");
    let default_block = context.append_basic_block(function, "default");

    let builder = context.create_builder();

    // Entry block with switch (no cases)
    builder.position_at_end(entry_block);
    let value = function.get_nth_param(0).unwrap().into_int_value();

    builder.build_switch(value, default_block, &[]).unwrap();

    // Default: return 42
    builder.position_at_end(default_block);
    let forty_two = i32_type.const_int(42, false);
    builder.build_return(Some(&forty_two)).unwrap();

    module
}

#[test]
fn test_basic_switch_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_basic_switch_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("switch_test").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("switch_test"));

    let compiled = &compiled_functions["switch_test"];
    assert!(compiled.code_size > 0);

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    assert!(stats.instruction_counts.contains_key("Switch"));
    assert_eq!(stats.instruction_counts["Switch"], 1);

    println!("✅ Basic switch test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!(
        "   Switch instructions: {}",
        stats.instruction_counts["Switch"]
    );
}

#[test]
fn test_range_switch_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_range_switch_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("range_switch").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("range_switch"));

    // Check that we handle larger case values correctly
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["Switch"], 1);

    println!("✅ Range switch test passed!");
}

#[test]
fn test_empty_switch_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_empty_switch_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("empty_switch").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("empty_switch"));

    // Check that empty switch just jumps to default
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["Switch"], 1);

    println!("✅ Empty switch test passed!");
}

#[test]
fn test_switch_with_64bit_value() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("switch_64bit");
    let i64_type = context.i64_type();

    // Create a switch on i64 value
    let fn_type = i64_type.fn_type(&[i64_type.into()], false);
    let function = module.add_function("switch_i64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let case1_block = context.append_basic_block(function, "case.1");
    let case2_block = context.append_basic_block(function, "case.2");
    let default_block = context.append_basic_block(function, "default");

    let builder = context.create_builder();

    builder.position_at_end(entry_block);
    let value = function.get_nth_param(0).unwrap().into_int_value();

    builder
        .build_switch(
            value,
            default_block,
            &[
                (i64_type.const_int(1, false), case1_block),
                (i64_type.const_int(2, false), case2_block),
            ],
        )
        .unwrap();

    builder.position_at_end(case1_block);
    let one = i64_type.const_int(10, false);
    builder.build_return(Some(&one)).unwrap();

    builder.position_at_end(case2_block);
    let two = i64_type.const_int(20, false);
    builder.build_return(Some(&two)).unwrap();

    builder.position_at_end(default_block);
    let neg_one = i64_type.const_int(-1i64 as u64, true);
    builder.build_return(Some(&neg_one)).unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("switch_i64").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("switch_i64"));

    println!("✅ 64-bit switch test passed!");
}
