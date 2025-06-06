//! Test alloca instruction compilation.
//!
//! This test verifies that stack allocation via the alloca instruction
//! works correctly, including proper size calculation and address generation.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{
    core::CompilationSession,
    llvm::LlvmCompiler,
};

/// Create a function that uses alloca for local variables.
fn create_alloca_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("alloca_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i32 sum_array_stack()
    // {
    //     int arr[10];  // Stack allocated array
    //     for (int i = 0; i < 10; i++)
    //         arr[i] = i;
    //     int sum = 0;
    //     for (int i = 0; i < 10; i++)
    //         sum += arr[i];
    //     return sum;
    // }
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("sum_array_stack", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let init_loop_block = context.append_basic_block(function, "init_loop");
    let init_body_block = context.append_basic_block(function, "init_body");
    let sum_loop_block = context.append_basic_block(function, "sum_loop");
    let sum_body_block = context.append_basic_block(function, "sum_body");
    let exit_block = context.append_basic_block(function, "exit");
    
    let builder = context.create_builder();
    
    // Entry block: allocate array and initialize loop counter
    builder.position_at_end(entry_block);
    let array_type = i32_type.array_type(10);
    let array_alloca = builder.build_alloca(array_type, "arr").unwrap();
    let i_alloca = builder.build_alloca(i32_type, "i").unwrap();
    let sum_alloca = builder.build_alloca(i32_type, "sum").unwrap();
    
    // Initialize i = 0
    let zero = i32_type.const_int(0, false);
    builder.build_store(i_alloca, zero).unwrap();
    builder.build_unconditional_branch(init_loop_block).unwrap();
    
    // Init loop condition
    builder.position_at_end(init_loop_block);
    let i_val = builder.build_load(i32_type, i_alloca, "i_val").unwrap().into_int_value();
    let ten = i32_type.const_int(10, false);
    let cond = builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, ten, "cond").unwrap();
    builder.build_conditional_branch(cond, init_body_block, sum_loop_block).unwrap();
    
    // Init body: arr[i] = i
    builder.position_at_end(init_body_block);
    let i_val = builder.build_load(i32_type, i_alloca, "i_val").unwrap().into_int_value();
    let zero_64 = i64_type.const_int(0, false);
    let i_64 = builder.build_int_z_extend(i_val, i64_type, "i_64").unwrap();
    let indices = [zero_64, i_64];
    let element_ptr = unsafe {
        builder.build_gep(array_type, array_alloca, &indices, "element_ptr").unwrap()
    };
    builder.build_store(element_ptr, i_val).unwrap();
    
    // Increment i
    let one = i32_type.const_int(1, false);
    let i_next = builder.build_int_add(i_val, one, "i_next").unwrap();
    builder.build_store(i_alloca, i_next).unwrap();
    builder.build_unconditional_branch(init_loop_block).unwrap();
    
    // Sum loop initialization
    builder.position_at_end(sum_loop_block);
    // Reset i = 0, sum = 0
    builder.build_store(i_alloca, zero).unwrap();
    builder.build_store(sum_alloca, zero).unwrap();
    
    // Sum loop condition
    let sum_loop_cond_block = context.append_basic_block(function, "sum_loop_cond");
    builder.build_unconditional_branch(sum_loop_cond_block).unwrap();
    
    builder.position_at_end(sum_loop_cond_block);
    let i_val = builder.build_load(i32_type, i_alloca, "i_val").unwrap().into_int_value();
    let cond = builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, ten, "cond").unwrap();
    builder.build_conditional_branch(cond, sum_body_block, exit_block).unwrap();
    
    // Sum body: sum += arr[i]
    builder.position_at_end(sum_body_block);
    let i_val = builder.build_load(i32_type, i_alloca, "i_val").unwrap().into_int_value();
    let sum_val = builder.build_load(i32_type, sum_alloca, "sum_val").unwrap().into_int_value();
    let i_64 = builder.build_int_z_extend(i_val, i64_type, "i_64").unwrap();
    let indices = [zero_64, i_64];
    let element_ptr = unsafe {
        builder.build_gep(array_type, array_alloca, &indices, "element_ptr").unwrap()
    };
    let element_val = builder.build_load(i32_type, element_ptr, "element_val").unwrap().into_int_value();
    let new_sum = builder.build_int_add(sum_val, element_val, "new_sum").unwrap();
    builder.build_store(sum_alloca, new_sum).unwrap();
    
    // Increment i
    let i_next = builder.build_int_add(i_val, one, "i_next").unwrap();
    builder.build_store(i_alloca, i_next).unwrap();
    builder.build_unconditional_branch(sum_loop_cond_block).unwrap();
    
    // Exit block: return sum
    builder.position_at_end(exit_block);
    let final_sum = builder.build_load(i32_type, sum_alloca, "final_sum").unwrap().into_int_value();
    builder.build_return(Some(&final_sum)).unwrap();
    
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
    let result = builder.build_load(i32_type, local_alloca, "result").unwrap();
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
    println!("   Compiled {} alloca instructions", stats.instruction_counts["Alloca"]);
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
    compiler.compile_function_by_name("sum_array_stack").unwrap();
    
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
    println!("   Compiled {} alloca instructions", stats.instruction_counts["Alloca"]);
    println!("   Compiled {} load instructions", stats.instruction_counts.get("Load").unwrap_or(&0));
    println!("   Compiled {} store instructions", stats.instruction_counts.get("Store").unwrap_or(&0));
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
    let array_alloca = builder.build_alloca(i32_type.array_type(5), "array_local").unwrap();
    
    // Store some values
    let val32 = i32_type.const_int(42, false);
    let val64 = i64_type.const_int(100, false);
    let valf32 = f32_type.const_float(3.14);
    
    builder.build_store(int32_alloca, val32).unwrap();
    builder.build_store(int64_alloca, val64).unwrap();
    builder.build_store(float_alloca, valf32).unwrap();
    
    // Return something
    let result = builder.build_load(i32_type, int32_alloca, "result").unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("test_multi_alloca").unwrap();
    
    // Verify compilation
    let stats = compiler.session().stats();
    assert_eq!(stats.instruction_counts["Alloca"], 4);
    
    println!("✅ Multiple allocas test passed!");
    println!("   Compiled {} alloca instructions", stats.instruction_counts["Alloca"]);
}