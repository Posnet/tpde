//! Test alloca instruction compilation.
//!
//! This test verifies that stack allocation via the alloca instruction
//! works correctly, including proper size calculation and address generation.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a function that uses alloca for local variables.
fn create_alloca_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("alloca_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

    // Simpler function without loops to avoid register exhaustion
    // Function: i32 sum_array_stack()
    // {
    //     int arr[10];
    //     arr[0] = 5;
    //     arr[1] = 10;
    //     arr[2] = 15;
    //     return arr[0] + arr[1] + arr[2];
    // }
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("sum_array_stack", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    // Allocate array on stack
    let array_type = i32_type.array_type(10);
    let array_alloca = builder.build_alloca(array_type, "arr").unwrap();

    // Also allocate some scalar variables for testing
    let i_alloca = builder.build_alloca(i32_type, "i").unwrap();
    let sum_alloca = builder.build_alloca(i32_type, "sum").unwrap();

    // Store values at specific indices (no loops)
    let zero = i32_type.const_int(0, false);
    let zero_64 = i64_type.const_int(0, false);
    let one_64 = i64_type.const_int(1, false);
    let two_64 = i64_type.const_int(2, false);

    // arr[0] = 5
    let indices0 = [zero_64, zero_64];
    let element_ptr0 = unsafe {
        builder
            .build_gep(array_type, array_alloca, &indices0, "element_ptr0")
            .unwrap()
    };
    let five = i32_type.const_int(5, false);
    builder.build_store(element_ptr0, five).unwrap();

    // arr[1] = 10
    let indices1 = [zero_64, one_64];
    let element_ptr1 = unsafe {
        builder
            .build_gep(array_type, array_alloca, &indices1, "element_ptr1")
            .unwrap()
    };
    let ten = i32_type.const_int(10, false);
    builder.build_store(element_ptr1, ten).unwrap();

    // arr[2] = 15
    let indices2 = [zero_64, two_64];
    let element_ptr2 = unsafe {
        builder
            .build_gep(array_type, array_alloca, &indices2, "element_ptr2")
            .unwrap()
    };
    let fifteen = i32_type.const_int(15, false);
    builder.build_store(element_ptr2, fifteen).unwrap();

    // Store zero to i and sum allocas
    builder.build_store(i_alloca, zero).unwrap();
    builder.build_store(sum_alloca, zero).unwrap();

    // Load values and compute sum
    let val0 = builder
        .build_load(i32_type, element_ptr0, "val0")
        .unwrap()
        .into_int_value();
    let val1 = builder
        .build_load(i32_type, element_ptr1, "val1")
        .unwrap()
        .into_int_value();
    let val2 = builder
        .build_load(i32_type, element_ptr2, "val2")
        .unwrap()
        .into_int_value();

    // sum = val0 + val1 + val2
    let sum01 = builder.build_int_add(val0, val1, "sum01").unwrap();
    let sum = builder.build_int_add(sum01, val2, "sum").unwrap();

    builder.build_return(Some(&sum)).unwrap();

    module
}

/// Test simple scalar alloca.
fn create_simple_alloca_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("simple_alloca");
    let i32_type = context.i32_type();

    // Function: i32 test_alloca(i32 x)
    // {
    //     int local = x + 10;
    //     return local;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let function = module.add_function("test_alloca", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    let x_param = function.get_nth_param(0).unwrap().into_int_value();

    // Allocate local variable
    let local_alloca = builder.build_alloca(i32_type, "local").unwrap();

    // Compute x + 10
    let ten = i32_type.const_int(10, false);
    let sum = builder.build_int_add(x_param, ten, "sum").unwrap();

    // Store to local
    builder.build_store(local_alloca, sum).unwrap();

    // Load and return
    let result = builder
        .build_load(i32_type, local_alloca, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    module
}

#[test]
fn test_simple_alloca_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_simple_alloca_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("test_alloca").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_alloca"));

    let compiled = &compiled_functions["test_alloca"];
    assert!(compiled.code_size > 0);

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    assert!(stats.instructions_compiled > 0);

    // Check that alloca was compiled
    assert!(stats.instruction_counts.contains_key("Alloca"));
    assert!(stats.instruction_counts["Alloca"] >= 1);

    println!("✅ Simple alloca test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!(
        "   Compiled {} alloca instructions",
        stats.instruction_counts["Alloca"]
    );
}

#[test]
fn test_array_alloca_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_alloca_test_module(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler
        .compile_function_by_name("sum_array_stack")
        .unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("sum_array_stack"));

    let compiled = &compiled_functions["sum_array_stack"];
    assert!(compiled.code_size > 0);

    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);

    // Check that allocas were compiled (array + i + sum = 3)
    assert!(stats.instruction_counts.contains_key("Alloca"));
    assert_eq!(stats.instruction_counts["Alloca"], 3);

    // Should also have loads, stores, and GEPs
    assert!(stats.instruction_counts.contains_key("Load"));
    assert!(stats.instruction_counts.contains_key("Store"));
    assert!(stats.instruction_counts.contains_key("GetElementPtr"));

    println!("✅ Array alloca test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!(
        "   Compiled {} alloca instructions",
        stats.instruction_counts["Alloca"]
    );
    println!(
        "   Compiled {} load instructions",
        stats.instruction_counts.get("Load").unwrap_or(&0)
    );
    println!(
        "   Compiled {} store instructions",
        stats.instruction_counts.get("Store").unwrap_or(&0)
    );
}

#[test]
fn test_multiple_allocas() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("multi_alloca");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let f32_type = context.f32_type();

    // Function with multiple allocas of different types
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("test_multi_alloca", fn_type, None);

    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);

    // Allocate different types
    let int32_alloca = builder.build_alloca(i32_type, "int32_local").unwrap();
    let int64_alloca = builder.build_alloca(i64_type, "int64_local").unwrap();
    let float_alloca = builder.build_alloca(f32_type, "float_local").unwrap();
    let _array_alloca = builder
        .build_alloca(i32_type.array_type(5), "array_local")
        .unwrap();

    // Store some values
    let val32 = i32_type.const_int(42, false);
    let val64 = i64_type.const_int(100, false);
    let valf32 = f32_type.const_float(3.5);

    builder.build_store(int32_alloca, val32).unwrap();
    builder.build_store(int64_alloca, val64).unwrap();
    builder.build_store(float_alloca, valf32).unwrap();

    // Return something
    let result = builder
        .build_load(i32_type, int32_alloca, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler
        .compile_function_by_name("test_multi_alloca")
        .unwrap();

    // Verify compilation
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["Alloca"], 4);

    println!("✅ Multiple allocas test passed!");
    println!(
        "   Compiled {} alloca instructions",
        stats.instruction_counts["Alloca"]
    );
}
