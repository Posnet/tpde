//! Test integer conversion operations (SEXT, ZEXT, TRUNC) compilation.
//!
//! This test verifies that integer type conversions are correctly compiled,
//! including sign extension, zero extension, and truncation.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{
    core::CompilationSession,
    llvm::LlvmCompiler,
};

/// Create a test module with sign extension operations.
fn create_sext_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("sext_test");
    let i8_type = context.i8_type();
    let i16_type = context.i16_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i32 test_sext_i8_to_i32(i8 value)
    // {
    //     return (i32)value;  // sign extend
    // }
    let fn_type = i32_type.fn_type(&[i8_type.into()], false);
    let function = module.add_function("test_sext_i8_to_i32", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_s_extend(value, i32_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Function: i64 test_sext_i16_to_i64(i16 value)
    // {
    //     return (i64)value;  // sign extend
    // }
    let fn_type = i64_type.fn_type(&[i16_type.into()], false);
    let function = module.add_function("test_sext_i16_to_i64", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_s_extend(value, i64_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Function: i64 test_sext_i32_to_i64(i32 value)
    // {
    //     return (i64)value;  // sign extend
    // }
    let fn_type = i64_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("test_sext_i32_to_i64", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_s_extend(value, i64_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    module
}

/// Create a test module with zero extension operations.
fn create_zext_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("zext_test");
    let i1_type = context.bool_type();
    let i8_type = context.i8_type();
    let _i16_type = context.i16_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i32 test_zext_i8_to_i32(i8 value)
    // {
    //     return (u32)value;  // zero extend
    // }
    let fn_type = i32_type.fn_type(&[i8_type.into()], false);
    let function = module.add_function("test_zext_i8_to_i32", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_z_extend(value, i32_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Function: i64 test_zext_i32_to_i64(i32 value)
    // {
    //     return (u64)value;  // zero extend
    // }
    let fn_type = i64_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("test_zext_i32_to_i64", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_z_extend(value, i64_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Function: i32 test_zext_i1_to_i32(i1 value)
    // {
    //     return value ? 1 : 0;  // zero extend boolean
    // }
    let fn_type = i32_type.fn_type(&[i1_type.into()], false);
    let function = module.add_function("test_zext_i1_to_i32", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_z_extend(value, i32_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    module
}

/// Create a test module with truncation operations.
fn create_trunc_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("trunc_test");
    let i1_type = context.bool_type();
    let i8_type = context.i8_type();
    let _i16_type = context.i16_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i32 test_trunc_i64_to_i32(i64 value)
    // {
    //     return (i32)value;  // truncate
    // }
    let fn_type = i32_type.fn_type(&[i64_type.into()], false);
    let function = module.add_function("test_trunc_i64_to_i32", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_truncate(value, i32_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Function: i8 test_trunc_i32_to_i8(i32 value)
    // {
    //     return (i8)value;  // truncate
    // }
    let fn_type = i8_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("test_trunc_i32_to_i8", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_truncate(value, i8_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Function: i1 test_trunc_i32_to_i1(i32 value)
    // {
    //     return value & 1;  // truncate to boolean
    // }
    let fn_type = i1_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("test_trunc_i32_to_i1", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let result = builder.build_int_truncate(value, i1_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    module
}

/// Create a complex test with multiple conversions.
fn create_complex_conversion_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("complex_conversion_test");
    let i8_type = context.i8_type();
    let i16_type = context.i16_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i64 sign_extend_chain(i8 value)
    // {
    //     i16 temp16 = (i16)value;  // sext i8 to i16
    //     i32 temp32 = (i32)temp16;  // sext i16 to i32
    //     return (i64)temp32;        // sext i32 to i64
    // }
    let fn_type = i64_type.fn_type(&[i8_type.into()], false);
    let function = module.add_function("sign_extend_chain", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let value = function.get_nth_param(0).unwrap().into_int_value();
    let temp16 = builder.build_int_s_extend(value, i16_type, "temp16").unwrap();
    let temp32 = builder.build_int_s_extend(temp16, i32_type, "temp32").unwrap();
    let result = builder.build_int_s_extend(temp32, i64_type, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    module
}

#[test]
fn test_sext_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    let context = Context::create();
    let module = create_sext_test(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile all functions
    compiler.compile_function_by_name("test_sext_i8_to_i32").unwrap();
    compiler.compile_function_by_name("test_sext_i16_to_i64").unwrap();
    compiler.compile_function_by_name("test_sext_i32_to_i64").unwrap();
    
    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_sext_i8_to_i32"));
    assert!(compiled_functions.contains_key("test_sext_i16_to_i64"));
    assert!(compiled_functions.contains_key("test_sext_i32_to_i64"));
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 3);
    assert!(stats.instruction_counts.contains_key("SExt"));
    assert_eq!(stats.instruction_counts["SExt"], 3);
    
    println!("✅ Sign extension test passed!");
    println!("   Compiled {} SEXT instructions", stats.instruction_counts["SExt"]);
}

#[test]
fn test_zext_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    let context = Context::create();
    let module = create_zext_test(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile all functions
    compiler.compile_function_by_name("test_zext_i8_to_i32").unwrap();
    compiler.compile_function_by_name("test_zext_i32_to_i64").unwrap();
    compiler.compile_function_by_name("test_zext_i1_to_i32").unwrap();
    
    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_zext_i8_to_i32"));
    assert!(compiled_functions.contains_key("test_zext_i32_to_i64"));
    assert!(compiled_functions.contains_key("test_zext_i1_to_i32"));
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 3);
    assert!(stats.instruction_counts.contains_key("ZExt"));
    assert_eq!(stats.instruction_counts["ZExt"], 3);
    
    println!("✅ Zero extension test passed!");
    println!("   Compiled {} ZEXT instructions", stats.instruction_counts["ZExt"]);
}

#[test]
fn test_trunc_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    let context = Context::create();
    let module = create_trunc_test(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile all functions
    compiler.compile_function_by_name("test_trunc_i64_to_i32").unwrap();
    compiler.compile_function_by_name("test_trunc_i32_to_i8").unwrap();
    compiler.compile_function_by_name("test_trunc_i32_to_i1").unwrap();
    
    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_trunc_i64_to_i32"));
    assert!(compiled_functions.contains_key("test_trunc_i32_to_i8"));
    assert!(compiled_functions.contains_key("test_trunc_i32_to_i1"));
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 3);
    assert!(stats.instruction_counts.contains_key("Trunc"));
    assert_eq!(stats.instruction_counts["Trunc"], 3);
    
    println!("✅ Truncation test passed!");
    println!("   Compiled {} TRUNC instructions", stats.instruction_counts["Trunc"]);
}

#[test]
fn test_complex_conversion_chain() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    let context = Context::create();
    let module = create_complex_conversion_test(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("sign_extend_chain").unwrap();
    
    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("sign_extend_chain"));
    
    let compiled = &compiled_functions["sign_extend_chain"];
    assert!(compiled.code_size > 0);
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    
    // Should have three SEXT instructions
    assert!(stats.instruction_counts.contains_key("SExt"));
    assert_eq!(stats.instruction_counts["SExt"], 3);
    
    println!("✅ Complex conversion chain test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!("   SEXT: {}", stats.instruction_counts["SExt"]);
}

#[test]
fn test_mixed_conversions() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    let context = Context::create();
    let module = context.create_module("mixed_conversions");
    let i16_type = context.i16_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i32 mixed_convert(i64 x, i16 y)
    // {
    //     i32 truncated = (i32)x;    // trunc
    //     i32 extended = (i32)y;     // sext
    //     return truncated + extended;
    // }
    let fn_type = i32_type.fn_type(&[i64_type.into(), i16_type.into()], false);
    let function = module.add_function("mixed_convert", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let x = function.get_nth_param(0).unwrap().into_int_value();
    let y = function.get_nth_param(1).unwrap().into_int_value();
    
    let truncated = builder.build_int_truncate(x, i32_type, "truncated").unwrap();
    let extended = builder.build_int_s_extend(y, i32_type, "extended").unwrap();
    let result = builder.build_int_add(truncated, extended, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("mixed_convert").unwrap();
    
    // Verify compilation
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["Trunc"], 1);
    assert_eq!(stats.instruction_counts["SExt"], 1);
    assert_eq!(stats.instruction_counts["Add"], 1);
    
    println!("✅ Mixed conversions test passed!");
}