//! Test shift operations (SHL, LSHR, ASHR) compilation.
//!
//! This test verifies that shift operations are correctly compiled,
//! including proper handling of the shift count in RCX register.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a test module with shift left operations.
fn create_shift_left_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("shift_left_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_shl32(i32 value, i32 shift)
    // {
    //     return value << shift;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_shl32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_left_shift(value, shift, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_shl64(i64 value, i64 shift)
    // {
    //     return value << shift;
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_shl64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_left_shift(value, shift, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a test module with logical shift right operations.
fn create_logical_shift_right_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("logical_shift_right_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_lshr32(i32 value, i32 shift)
    // {
    //     return value >>> shift;  // logical shift right
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_lshr32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder
        .build_right_shift(value, shift, false, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_lshr64(i64 value, i64 shift)
    // {
    //     return value >>> shift;  // logical shift right
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_lshr64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder
        .build_right_shift(value, shift, false, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a test module with arithmetic shift right operations.
fn create_arithmetic_shift_right_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("arithmetic_shift_right_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_ashr32(i32 value, i32 shift)
    // {
    //     return value >> shift;  // arithmetic shift right
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_ashr32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder
        .build_right_shift(value, shift, true, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_ashr64(i64 value, i64 shift)
    // {
    //     return value >> shift;  // arithmetic shift right
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_ashr64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder
        .build_right_shift(value, shift, true, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a complex test with multiple shift operations.
fn create_complex_shift_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("complex_shift_test");
    let i32_type = context.i32_type();

    // Function: i32 rotate_left(i32 value, i32 count)
    // {
    //     // Simulated rotate left using shifts
    //     i32 left = value << count;
    //     i32 right = value >>> (32 - count);
    //     return left | right;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("rotate_left", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let count = function.get_nth_param(1).unwrap().into_int_value();

    // left = value << count
    let left = builder.build_left_shift(value, count, "left").unwrap();

    // right_count = 32 - count
    let thirty_two = i32_type.const_int(32, false);
    let right_count = builder
        .build_int_sub(thirty_two, count, "right_count")
        .unwrap();

    // right = value >>> right_count
    let right = builder
        .build_right_shift(value, right_count, false, "right")
        .unwrap();

    // result = left | right
    let result = builder.build_or(left, right, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

#[test]
fn test_shift_left_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_shift_left_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_shl32").unwrap();
    compiler.compile_function_by_name("test_shl64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_shl32"));
    assert!(compiled_functions.contains_key("test_shl64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("Shl"));
    assert_eq!(stats.instruction_counts["Shl"], 2);

    println!("✅ Shift left test passed!");
    println!(
        "   Compiled {} SHL instructions",
        stats.instruction_counts["Shl"]
    );
}

#[test]
fn test_logical_shift_right_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_logical_shift_right_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_lshr32").unwrap();
    compiler.compile_function_by_name("test_lshr64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_lshr32"));
    assert!(compiled_functions.contains_key("test_lshr64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("LShr"));
    assert_eq!(stats.instruction_counts["LShr"], 2);

    println!("✅ Logical shift right test passed!");
    println!(
        "   Compiled {} LSHR instructions",
        stats.instruction_counts["LShr"]
    );
}

#[test]
fn test_arithmetic_shift_right_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_arithmetic_shift_right_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_ashr32").unwrap();
    compiler.compile_function_by_name("test_ashr64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_ashr32"));
    assert!(compiled_functions.contains_key("test_ashr64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("AShr"));
    assert_eq!(stats.instruction_counts["AShr"], 2);

    println!("✅ Arithmetic shift right test passed!");
    println!(
        "   Compiled {} ASHR instructions",
        stats.instruction_counts["AShr"]
    );
}

#[test]
fn test_complex_shift_operations() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_complex_shift_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("rotate_left").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("rotate_left"));

    let compiled = &compiled_functions["rotate_left"];
    assert!(compiled.code_size > 0);

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);

    // Should have both SHL and LSHR operations
    assert!(stats.instruction_counts.contains_key("Shl"));
    assert!(stats.instruction_counts.contains_key("LShr"));
    assert_eq!(stats.instruction_counts["Shl"], 1);
    assert_eq!(stats.instruction_counts["LShr"], 1);

    println!("✅ Complex shift operations test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!(
        "   SHL: {}, LSHR: {}",
        stats.instruction_counts["Shl"], stats.instruction_counts["LShr"]
    );
}

#[test]
fn test_shift_with_constants() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("shift_constants");
    let i32_type = context.i32_type();

    // Function: i32 shift_by_constant(i32 value)
    // {
    //     return value << 4;  // Shift by constant
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("shift_by_constant", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let shift_amount = i32_type.const_int(4, false);
    let result = builder
        .build_left_shift(value, shift_amount, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler
        .compile_function_by_name("shift_by_constant")
        .unwrap();

    // Verify function was compiled
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["Shl"], 1);

    println!("✅ Shift with constants test passed!");
}
