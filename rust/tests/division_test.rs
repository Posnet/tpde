//! Test integer division and remainder operations compilation.
//!
//! This test verifies that division operations are correctly compiled,
//! including signed/unsigned division and remainder operations.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a test module with unsigned division operations.
fn create_udiv_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("udiv_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_udiv32(i32 a, i32 b)
    // {
    //     return a / b;  // unsigned division
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_udiv32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_int_unsigned_div(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_udiv64(i64 a, i64 b)
    // {
    //     return a / b;  // unsigned division
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_udiv64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_int_unsigned_div(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a test module with signed division operations.
fn create_sdiv_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("sdiv_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Function: i32 test_sdiv32(i32 a, i32 b)
    // {
    //     return a / b;  // signed division
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_sdiv32", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_int_signed_div(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i64 test_sdiv64(i64 a, i64 b)
    // {
    //     return a / b;  // signed division
    // }
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("test_sdiv64", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_int_signed_div(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a test module with remainder operations.
fn create_rem_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("rem_test");
    let i32_type = context.i32_type();

    // Function: i32 test_urem(i32 a, i32 b)
    // {
    //     return a % b;  // unsigned remainder
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_urem", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_int_unsigned_rem(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Function: i32 test_srem(i32 a, i32 b)
    // {
    //     return a % b;  // signed remainder
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_srem", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let result = builder.build_int_signed_rem(a, b, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Create a complex test with division and remainder operations.
fn create_complex_division_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("complex_division_test");
    let i32_type = context.i32_type();

    // Function: i32 euclidean_algorithm(i32 a, i32 b)
    // {
    //     while (b != 0) {
    //         i32 temp = b;
    //         b = a % b;
    //         a = temp;
    //     }
    //     return a;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("euclidean_algorithm", fn_type, None);

    // Create basic blocks
    let entry = context.append_basic_block(function, "entry");
    let loop_header = context.append_basic_block(function, "loop_header");
    let loop_body = context.append_basic_block(function, "loop_body");
    let loop_exit = context.append_basic_block(function, "loop_exit");

    let builder = context.create_builder();

    // Entry block
    builder.position_at_end(entry);
    let a_param = function.get_nth_param(0).unwrap().into_int_value();
    let b_param = function.get_nth_param(1).unwrap().into_int_value();

    // Allocate variables
    let a_alloca = builder.build_alloca(i32_type, "a").unwrap();
    let b_alloca = builder.build_alloca(i32_type, "b").unwrap();
    builder.build_store(a_alloca, a_param).unwrap();
    builder.build_store(b_alloca, b_param).unwrap();
    builder.build_unconditional_branch(loop_header).unwrap();

    // Loop header: check if b != 0
    builder.position_at_end(loop_header);
    let b_val = builder
        .build_load(i32_type, b_alloca, "b_val")
        .unwrap()
        .into_int_value();
    let zero = i32_type.const_int(0, false);
    let cond = builder
        .build_int_compare(inkwell::IntPredicate::NE, b_val, zero, "cond")
        .unwrap();
    builder
        .build_conditional_branch(cond, loop_body, loop_exit)
        .unwrap();

    // Loop body: temp = b; b = a % b; a = temp;
    builder.position_at_end(loop_body);
    let a_val = builder
        .build_load(i32_type, a_alloca, "a_val")
        .unwrap()
        .into_int_value();
    let b_val2 = builder
        .build_load(i32_type, b_alloca, "b_val2")
        .unwrap()
        .into_int_value();

    // Calculate a % b
    let remainder = builder
        .build_int_signed_rem(a_val, b_val2, "remainder")
        .unwrap();

    // Update values: a = b, b = remainder
    builder.build_store(a_alloca, b_val2).unwrap();
    builder.build_store(b_alloca, remainder).unwrap();
    builder.build_unconditional_branch(loop_header).unwrap();

    // Loop exit: return a
    builder.position_at_end(loop_exit);
    let result = builder.build_load(i32_type, a_alloca, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

#[test]
fn test_udiv_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_udiv_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_udiv32").unwrap();
    compiler.compile_function_by_name("test_udiv64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_udiv32"));
    assert!(compiled_functions.contains_key("test_udiv64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("UDiv"));
    assert_eq!(stats.instruction_counts["UDiv"], 2);

    println!("✅ Unsigned division test passed!");
    println!(
        "   Compiled {} UDIV instructions",
        stats.instruction_counts["UDiv"]
    );
}

#[test]
fn test_sdiv_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_sdiv_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_sdiv32").unwrap();
    compiler.compile_function_by_name("test_sdiv64").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_sdiv32"));
    assert!(compiled_functions.contains_key("test_sdiv64"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("SDiv"));
    assert_eq!(stats.instruction_counts["SDiv"], 2);

    println!("✅ Signed division test passed!");
    println!(
        "   Compiled {} SDIV instructions",
        stats.instruction_counts["SDiv"]
    );
}

#[test]
fn test_remainder_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_rem_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile both functions
    compiler.compile_function_by_name("test_urem").unwrap();
    compiler.compile_function_by_name("test_srem").unwrap();

    // Verify functions were compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_urem"));
    assert!(compiled_functions.contains_key("test_srem"));

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 2);
    assert!(stats.instruction_counts.contains_key("URem"));
    assert!(stats.instruction_counts.contains_key("SRem"));
    assert_eq!(stats.instruction_counts["URem"], 1);
    assert_eq!(stats.instruction_counts["SRem"], 1);

    println!("✅ Remainder operations test passed!");
    println!(
        "   UREM: {}, SREM: {}",
        stats.instruction_counts["URem"], stats.instruction_counts["SRem"]
    );
}

#[test]
fn test_complex_division_algorithm() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_complex_division_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler
        .compile_function_by_name("euclidean_algorithm")
        .unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("euclidean_algorithm"));

    let compiled = &compiled_functions["euclidean_algorithm"];
    assert!(compiled.code_size > 0);

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);

    // Should have remainder instruction for the modulo operation
    assert!(stats.instruction_counts.contains_key("SRem"));
    assert!(stats.instruction_counts["SRem"] >= 1);

    println!("✅ Complex division algorithm test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!(
        "   SREM: {}",
        stats.instruction_counts.get("SRem").unwrap_or(&0)
    );
}

#[test]
fn test_division_with_constants() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("const_division");
    let i32_type = context.i32_type();

    // Function: i32 divide_by_constant(i32 value)
    // {
    //     return value / 10;  // Division by constant
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("divide_by_constant", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let value = function.get_nth_param(0).unwrap().into_int_value();
    let ten = i32_type.const_int(10, false);
    let result = builder.build_int_signed_div(value, ten, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler
        .compile_function_by_name("divide_by_constant")
        .unwrap();

    // Verify compilation
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["SDiv"], 1);

    println!("✅ Division by constant test passed!");
}
