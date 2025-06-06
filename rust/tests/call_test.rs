//! Test function call compilation.

use tpde::{CompilationSession, LlvmCompiler};
use inkwell::context::Context;
use inkwell::IntPredicate;
use bumpalo::Bump;

/// Create a test module with function calls.
fn create_module_with_calls(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("call_test");
    let i32_type = context.i32_type();
    
    // Create a simple function to call: int add(int a, int b)
    let add_fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let add_function = module.add_function("add", add_fn_type, None);
    
    let add_entry = context.append_basic_block(add_function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(add_entry);
    
    let a = add_function.get_nth_param(0).unwrap().into_int_value();
    let b = add_function.get_nth_param(1).unwrap().into_int_value();
    let sum = builder.build_int_add(a, b, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();
    
    // Create a function that calls add: int test_call(int x, int y)
    let test_fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let test_function = module.add_function("test_call", test_fn_type, None);
    
    let test_entry = context.append_basic_block(test_function, "entry");
    builder.position_at_end(test_entry);
    
    let x = test_function.get_nth_param(0).unwrap().into_int_value();
    let y = test_function.get_nth_param(1).unwrap().into_int_value();
    
    // Call add(x, y)
    let call_result = builder.build_call(add_function, &[x.into(), y.into()], "result").unwrap();
    let result = call_result.try_as_basic_value().left().unwrap().into_int_value();
    
    builder.build_return(Some(&result)).unwrap();
    
    module
}

/// Create a recursive factorial function.
fn create_factorial_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("factorial_test");
    let i32_type = context.i32_type();
    
    // Create factorial function: int factorial(int n)
    let fact_fn_type = i32_type.fn_type(&[i32_type.into()], false);
    let fact_function = module.add_function("factorial", fact_fn_type, None);
    
    // Create basic blocks
    let entry = context.append_basic_block(fact_function, "entry");
    let base_case = context.append_basic_block(fact_function, "base_case");
    let recursive_case = context.append_basic_block(fact_function, "recursive_case");
    
    let builder = context.create_builder();
    
    // Entry block: check if n <= 1
    builder.position_at_end(entry);
    let n = fact_function.get_nth_param(0).unwrap().into_int_value();
    let one = i32_type.const_int(1, false);
    let cond = builder.build_int_compare(IntPredicate::SLE, n, one, "cond").unwrap();
    builder.build_conditional_branch(cond, base_case, recursive_case).unwrap();
    
    // Base case: return 1
    builder.position_at_end(base_case);
    builder.build_return(Some(&one)).unwrap();
    
    // Recursive case: return n * factorial(n - 1)
    builder.position_at_end(recursive_case);
    let n_minus_1 = builder.build_int_sub(n, one, "n_minus_1").unwrap();
    let call_result = builder.build_call(fact_function, &[n_minus_1.into()], "call_result").unwrap();
    let recursive_result = call_result.try_as_basic_value().left().unwrap().into_int_value();
    let product = builder.build_int_mul(n, recursive_result, "product").unwrap();
    builder.build_return(Some(&product)).unwrap();
    
    module
}

#[test]
fn test_simple_function_call() {
    let context = Context::create();
    let module = create_module_with_calls(&context);
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the test_call function
    let result = compiler.compile_function_by_name("test_call");
    assert!(result.is_ok(), "Failed to compile test_call: {:?}", result);
    
    // Check statistics
    let stats = session.stats();
    assert!(stats.total_calls > 0, "Should have recorded at least one call");
    assert!(stats.instructions_compiled > 0);
    
    println!("✅ Simple function call test passed");
}

#[test]
fn test_recursive_function_call() {
    let context = Context::create();
    let module = create_factorial_module(&context);
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the factorial function
    let result = compiler.compile_function_by_name("factorial");
    assert!(result.is_ok(), "Failed to compile factorial: {:?}", result);
    
    // Check statistics
    let stats = session.stats();
    assert!(stats.total_calls > 0, "Should have recorded at least one recursive call");
    assert!(stats.instructions_compiled > 0);
    
    // Should have compiled branches, comparisons, arithmetic, and calls
    let counts = &stats.instruction_counts;
    assert!(counts.contains_key("Call"), "Should have Call instructions");
    assert!(counts.contains_key("ICmp"), "Should have ICmp instructions");
    assert!(counts.contains_key("Br"), "Should have Br instructions");
    
    println!("✅ Recursive function call test passed");
}

#[test]
fn test_multiple_arguments() {
    let context = Context::create();
    let module = context.create_module("multi_arg_test");
    let i32_type = context.i32_type();
    
    // Create a function with many arguments to test register and stack passing
    // int many_args(int a, int b, int c, int d, int e, int f, int g, int h)
    let arg_types = vec![i32_type.into(); 8];
    let many_args_type = i32_type.fn_type(&arg_types, false);
    let many_args_fn = module.add_function("many_args", many_args_type, None);
    
    let entry = context.append_basic_block(many_args_fn, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);
    
    // Just return the sum of first and last argument
    let first = many_args_fn.get_nth_param(0).unwrap().into_int_value();
    let last = many_args_fn.get_nth_param(7).unwrap().into_int_value();
    let sum = builder.build_int_add(first, last, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();
    
    // Create a caller function
    let caller_type = i32_type.fn_type(&[], false);
    let caller_fn = module.add_function("caller", caller_type, None);
    
    let caller_entry = context.append_basic_block(caller_fn, "entry");
    builder.position_at_end(caller_entry);
    
    // Call with 8 arguments (first 6 in registers, last 2 on stack)
    let args: Vec<_> = (1..=8).map(|i| i32_type.const_int(i, false).into()).collect();
    let call_result = builder.build_call(many_args_fn, &args, "result").unwrap();
    let result = call_result.try_as_basic_value().left().unwrap();
    builder.build_return(Some(&result)).unwrap();
    
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the caller function
    let result = compiler.compile_function_by_name("caller");
    assert!(result.is_ok(), "Failed to compile caller: {:?}", result);
    
    println!("✅ Multiple arguments call test passed");
}

#[test]
fn test_void_function_call() {
    let context = Context::create();
    let module = context.create_module("void_test");
    let i32_type = context.i32_type();
    let void_type = context.void_type();
    
    // Create a void function: void do_nothing()
    let void_fn_type = void_type.fn_type(&[], false);
    let void_fn = module.add_function("do_nothing", void_fn_type, None);
    
    let void_entry = context.append_basic_block(void_fn, "entry");
    let builder = context.create_builder();
    builder.position_at_end(void_entry);
    builder.build_return(None).unwrap();
    
    // Create a function that calls the void function
    let caller_type = i32_type.fn_type(&[], false);
    let caller_fn = module.add_function("void_caller", caller_type, None);
    
    let caller_entry = context.append_basic_block(caller_fn, "entry");
    builder.position_at_end(caller_entry);
    
    // Call void function
    builder.build_call(void_fn, &[], "").unwrap();
    
    // Return a constant
    let const_val = i32_type.const_int(42, false);
    builder.build_return(Some(&const_val)).unwrap();
    
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();
    
    // Compile the caller function
    let result = compiler.compile_function_by_name("void_caller");
    assert!(result.is_ok(), "Failed to compile void_caller: {:?}", result);
    
    println!("✅ Void function call test passed");
}