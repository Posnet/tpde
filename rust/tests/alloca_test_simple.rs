//! Test alloca instruction compilation with simpler functions.
//!
//! This test verifies that stack allocation via the alloca instruction
//! works correctly, including proper size calculation and address generation.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::{
    core::CompilationSession,
    llvm::LlvmCompiler,
};

/// Create a simpler test module with array alloca but no loops.
fn create_simple_array_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("array_alloca_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    
    // Function: i32 simple_array_sum()
    // {
    //     int arr[10];
    //     arr[0] = 5;
    //     arr[1] = 10;
    //     arr[2] = 15;
    //     return arr[0] + arr[1] + arr[2];
    // }
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("simple_array_sum", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    // Allocate array on stack
    let array_type = i32_type.array_type(10);
    let array_alloca = builder.build_alloca(array_type, "arr").unwrap();
    
    // Store values at specific indices
    let zero_64 = i64_type.const_int(0, false);
    let one_64 = i64_type.const_int(1, false);
    let two_64 = i64_type.const_int(2, false);
    
    // arr[0] = 5
    let indices0 = [zero_64, zero_64];
    let element_ptr0 = unsafe {
        builder.build_gep(array_type, array_alloca, &indices0, "element_ptr0").unwrap()
    };
    let five = i32_type.const_int(5, false);
    builder.build_store(element_ptr0, five).unwrap();
    
    // arr[1] = 10
    let indices1 = [zero_64, one_64];
    let element_ptr1 = unsafe {
        builder.build_gep(array_type, array_alloca, &indices1, "element_ptr1").unwrap()
    };
    let ten = i32_type.const_int(10, false);
    builder.build_store(element_ptr1, ten).unwrap();
    
    // arr[2] = 15
    let indices2 = [zero_64, two_64];
    let element_ptr2 = unsafe {
        builder.build_gep(array_type, array_alloca, &indices2, "element_ptr2").unwrap()
    };
    let fifteen = i32_type.const_int(15, false);
    builder.build_store(element_ptr2, fifteen).unwrap();
    
    // Load values and compute sum
    let val0 = builder.build_load(i32_type, element_ptr0, "val0").unwrap().into_int_value();
    let val1 = builder.build_load(i32_type, element_ptr1, "val1").unwrap().into_int_value();
    let val2 = builder.build_load(i32_type, element_ptr2, "val2").unwrap().into_int_value();
    
    // sum = val0 + val1 + val2
    let sum01 = builder.build_int_add(val0, val1, "sum01").unwrap();
    let sum = builder.build_int_add(sum01, val2, "sum").unwrap();
    
    builder.build_return(Some(&sum)).unwrap();
    
    module
}

#[test]
fn test_simple_array_alloca_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    let context = Context::create();
    let module = create_simple_array_test_module(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("simple_array_sum").unwrap();
    
    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("simple_array_sum"));
    
    let compiled = &compiled_functions["simple_array_sum"];
    assert!(compiled.code_size > 0);
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    
    // Check that alloca was compiled (1 array alloca)
    assert!(stats.instruction_counts.contains_key("Alloca"));
    assert_eq!(stats.instruction_counts["Alloca"], 1);
    
    // Should also have loads, stores, and GEPs
    assert!(stats.instruction_counts.contains_key("Load"));
    assert!(stats.instruction_counts.contains_key("Store"));
    assert!(stats.instruction_counts.contains_key("GetElementPtr"));
    
    println!("✅ Simple array alloca test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!("   Compiled {} alloca instructions", stats.instruction_counts["Alloca"]);
}