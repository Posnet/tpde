//! Tests for the concrete LLVM compiler implementation.

use bumpalo::Bump;
use inkwell::context::Context;
use inkwell::IntPredicate;
use tpde::{
    compilation_session::CompilationSession,
    llvm_compiler_concrete::LlvmCompiler,
};

/// Create a simple test module with load/store operations.
fn create_load_store_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("load_store_test");
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    
    // Function: i32 load_add_store(i32* ptr, i32 value)
    // {
    //     i32 old = *ptr;
    //     *ptr = old + value;
    //     return old;
    // }
    let fn_type = i32_type.fn_type(&[ptr_type.into(), i32_type.into()], false);
    let function = module.add_function("load_add_store", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let ptr_param = function.get_nth_param(0).unwrap().into_pointer_value();
    let value_param = function.get_nth_param(1).unwrap().into_int_value();
    
    // Load old value
    let old_value = builder.build_load(i32_type, ptr_param, "old").unwrap().into_int_value();
    
    // Add value
    let new_value = builder.build_int_add(old_value, value_param, "new").unwrap();
    
    // Store new value
    builder.build_store(ptr_param, new_value).unwrap();
    
    // Return old value
    builder.build_return(Some(&old_value)).unwrap();
    
    module
}

/// Create a test module with conditional branches.
fn create_branch_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("branch_test");
    let i32_type = context.i32_type();
    
    // Function: i32 max(i32 a, i32 b)
    // {
    //     if (a > b)
    //         return a;
    //     else
    //         return b;
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("max", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let then_block = context.append_basic_block(function, "then");
    let else_block = context.append_basic_block(function, "else");
    
    let builder = context.create_builder();
    
    // Entry block
    builder.position_at_end(entry_block);
    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::SGT, a, b, "cmp").unwrap();
    builder.build_conditional_branch(cmp, then_block, else_block).unwrap();
    
    // Then block
    builder.position_at_end(then_block);
    builder.build_return(Some(&a)).unwrap();
    
    // Else block
    builder.position_at_end(else_block);
    builder.build_return(Some(&b)).unwrap();
    
    module
}

#[test]
fn test_load_store_compilation() {
    let context = Context::create();
    let module = create_load_store_test_module(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let mut session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &mut session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("load_add_store").unwrap();
    
    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("load_add_store"));
    
    let compiled = &compiled_functions["load_add_store"];
    assert!(compiled.code_size > 0);
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    assert!(stats.instructions_compiled > 0);
    
    println!("✅ Load/Store test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!("   Compiled {} instructions", stats.instructions_compiled);
}

#[test]
fn test_branch_compilation() {
    let context = Context::create();
    let module = create_branch_test_module(&context);
    
    // Create arena and session
    let arena = Bump::new();
    let mut session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &mut session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("max").unwrap();
    
    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("max"));
    
    let compiled = &compiled_functions["max"];
    assert!(compiled.code_size > 0);
    
    // Check session statistics
    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    assert!(stats.instructions_compiled > 0);
    
    // Check that we compiled branch instructions
    assert!(stats.instruction_counts.contains_key("Br"));
    
    println!("✅ Branch test passed!");
    println!("   Generated {} bytes of code", compiled.code_size);
    println!("   Compiled {} instructions", stats.instructions_compiled);
    println!("   Including {} branch instructions", stats.instruction_counts["Br"]);
}

#[test]
fn test_gep_compilation() {
    let context = Context::create();
    let module = context.create_module("gep_test");
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    
    // Function: i32* index_array(i32* arr, i32 idx)
    let fn_type = ptr_type.fn_type(&[ptr_type.into(), i32_type.into()], false);
    let function = module.add_function("index_array", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let arr_param = function.get_nth_param(0).unwrap().into_pointer_value();
    let idx_param = function.get_nth_param(1).unwrap().into_int_value();
    
    // GEP to calculate &arr[idx]
    let gep_indices = [idx_param];
    let element_ptr = unsafe {
        builder.build_gep(i32_type, arr_param, &gep_indices, "element_ptr").unwrap()
    };
    
    builder.build_return(Some(&element_ptr)).unwrap();
    
    // Create arena and session
    let arena = Bump::new();
    let mut session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &mut session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("index_array").unwrap();
    
    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("index_array"));
    
    // Check that we compiled GEP instruction
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("GetElementPtr"));
    
    println!("✅ GEP test passed!");
    println!("   Compiled {} GEP instructions", stats.instruction_counts["GetElementPtr"]);
}

#[test]
fn test_arithmetic_operations() {
    let context = Context::create();
    let module = context.create_module("arithmetic_test");
    let i32_type = context.i32_type();
    
    // Function: i32 arithmetic(i32 a, i32 b)
    // {
    //     return (a + b) * (a - b);
    // }
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("arithmetic", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    
    let sum = builder.build_int_add(a, b, "sum").unwrap();
    let diff = builder.build_int_sub(a, b, "diff").unwrap();
    let product = builder.build_int_mul(sum, diff, "product").unwrap();
    
    builder.build_return(Some(&product)).unwrap();
    
    // Create arena and session
    let arena = Bump::new();
    let mut session = CompilationSession::new(&arena);
    
    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &mut session).unwrap();
    
    // Compile the function
    compiler.compile_function_by_name("arithmetic").unwrap();
    
    // Check that we compiled arithmetic instructions
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("Add"));
    assert!(stats.instruction_counts.contains_key("Sub"));
    assert!(stats.instruction_counts.contains_key("Mul"));
    
    println!("✅ Arithmetic test passed!");
    println!("   Add: {}, Sub: {}, Mul: {}", 
             stats.instruction_counts["Add"],
             stats.instruction_counts["Sub"], 
             stats.instruction_counts["Mul"]);
}