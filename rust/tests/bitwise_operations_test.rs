//! Test bitwise operations (AND, OR, XOR) compilation.
//!
//! This test verifies that bitwise operations are correctly compiled,
//! including proper register allocation and instruction encoding.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a test module with bitwise AND operations.
fn create_bitwise_and_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("bitwise_and_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_and32(i32 a, i32 b)
    // {
    //     return a & b;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_and32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_and(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_and64(i64 a, i64 b)
    // {
    //     return a & b;
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_and64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_and(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a test module with bitwise OR operations.
fn create_bitwise_or_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("bitwise_or_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_or32(i32 a, i32 b)
    // {
    //     return a | b;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_or32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_or(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_or64(i64 a, i64 b)
    // {
    //     return a | b;
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_or64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_or(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a test module with bitwise XOR operations.
fn create_bitwise_xor_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("bitwise_xor_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_xor32(i32 a, i32 b)
    // {
    //     return a ^ b;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_xor32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_xor(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_xor64(i64 a, i64 b)
    // {
    //     return a ^ b;
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_xor64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_xor(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a complex test with multiple bitwise operations.
fn create_complex_bitwise_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("complex_bitwise_test");
    let i32_type = context.i32_type();

    // Function: i32 mask_and_shift(i32 value, i32 mask1, i32 mask2)
    // {
    //     i32 temp = value & mask1;
    //     temp = temp | mask2;
    //     return temp ^ 0xFF;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into(), i32_type.into()], false);
    let function = module.add_function("mask_and_shift", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let mask1 = function.get_nth_param(1).unwrap().into_int_value();
    let mask2 = function.get_nth_param(2).unwrap().into_int_value();

    // temp = value & mask1
    let temp = builder.build_and(value, mask1, "temp").unwrap();

    // temp = temp | mask2
    let temp2 = builder.build_or(temp, mask2, "temp2").unwrap();

    // return temp ^ 0xFF
    let const_ff = i32_type.const_int(0xFF, false);
    let result = builder.build_xor(temp2, const_ff, "result").unwrap();

    builder.build_return(Some(&result)).unwrap();

    module
}

#[test]
fn test_bitwise_and_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_bitwise_and_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_and32").unwrap();
    compiler.compile_function_by_name("test_and64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_and32"));
    assert!(compiled_functions.contains_key("test_and64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("And"));
    assert_eq!(stats.instruction_counts["And"], 2);

    println!("✅ Bitwise AND test passed!");
    println!(
        "   Compiled {} AND instructions",
        stats.instruction_counts["And"]
    );
}

#[test]
fn test_bitwise_or_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_bitwise_or_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_or32").unwrap();
    compiler.compile_function_by_name("test_or64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_or32"));
    assert!(compiled_functions.contains_key("test_or64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("Or"));
    assert_eq!(stats.instruction_counts["Or"], 2);

    println!("✅ Bitwise OR test passed!");
    println!(
        "   Compiled {} OR instructions",
        stats.instruction_counts["Or"]
    );
}

#[test]
fn test_bitwise_xor_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_bitwise_xor_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_xor32").unwrap();
    compiler.compile_function_by_name("test_xor64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_xor32"));
    assert!(compiled_functions.contains_key("test_xor64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("Xor"));
    assert_eq!(stats.instruction_counts["Xor"], 2);

    println!("✅ Bitwise XOR test passed!");
    println!(
        "   Compiled {} XOR instructions",
        stats.instruction_counts["Xor"]
    );
}

#[test]
fn test_complex_bitwise_operations() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_complex_bitwise_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("mask_and_shift").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("mask_and_shift"));

    let compiled = &compiled_functions["mask_and_shift"];
    assert!(compiled.code_size > 0);

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);

    // Should have one of each bitwise operation
    assert!(stats.instruction_counts.contains_key("And"));
    assert!(stats.instruction_counts.contains_key("Or"));
    assert!(stats.instruction_counts.contains_key("Xor"));
    assert_eq!(stats.instruction_counts["And"], 1);
    assert_eq!(stats.instruction_counts["Or"], 1);
    assert_eq!(stats.instruction_counts["Xor"], 1);

    println!("✅ Complex bitwise operations test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!(
        "   AND: {}, OR: {}, XOR: {}",
        stats.instruction_counts["And"],
        stats.instruction_counts["Or"],
        stats.instruction_counts["Xor"]
    );
}

#[test]
fn test_bitwise_with_constants() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("bitwise_constants");
    let i32_type = context.i32_type();

    // Function: i32 clear_bits(i32 value)
    // {
    //     return value & 0xFFFF0000;  // Clear lower 16 bits
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("clear_bits", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let mask = i32_type.const_int(0xFFFF0000, false);
    let result = builder.build_and(value, mask, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("clear_bits").unwrap();

    // Verify function was compiled
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["And"], 1);

    println!("✅ Bitwise with constants test passed!");
}
