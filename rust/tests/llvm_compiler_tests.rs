// This test suite comprehensively validates the concrete LlvmCompiler implementation after
// the architectural redesign that eliminated trait bound complexity. The tests cover core
// compilation scenarios including load/store operations for memory access patterns, conditional
// branches for control flow (if/else), GEP instructions for array indexing, and arithmetic
// operations (add, sub, mul). Each test creates a minimal LLVM IR function using inkwell,
// compiles it with LlvmCompiler, and verifies successful code generation by checking that
// machine code was produced (non-zero size), functions were tracked in compiled_functions map,
// session statistics were updated correctly, and specific instruction types were recognized
// and compiled. These integration tests validate the direct LLVM integration approach where
// InstructionValue<'ctx> is used throughout without abstraction layers. The tests demonstrate
// that the simplified architecture successfully compiles real LLVM IR patterns while maintaining
// the performance and functionality goals of TPDE. They serve as regression tests to ensure
// core compilation capabilities remain intact as the compiler evolves.

//! Tests for the concrete LLVM compiler implementation.

use bumpalo::Bump;
use inkwell::context::Context;
use inkwell::IntPredicate;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

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
    let old_value = builder
        .build_load(i32_type, ptr_param, "old")
        .unwrap()
        .into_int_value();

    // Add value
    let new_value = builder
        .build_int_add(old_value, value_param, "new")
        .unwrap();

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
    let cmp = builder
        .build_int_compare(IntPredicate::SGT, a, b, "cmp")
        .unwrap();
    builder
        .build_conditional_branch(cmp, then_block, else_block)
        .unwrap();

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
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

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
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

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
    println!(
        "   Including {} branch instructions",
        stats.instruction_counts["Br"]
    );
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
        builder
            .build_gep(i32_type, arr_param, &gep_indices, "element_ptr")
            .unwrap()
    };

    builder.build_return(Some(&element_ptr)).unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("index_array").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("index_array"));

    // Check that we compiled GEP instruction
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("GetElementPtr"));

    println!("✅ GEP test passed!");
    println!(
        "   Compiled {} GEP instructions",
        stats.instruction_counts["GetElementPtr"]
    );
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
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("arithmetic").unwrap();

    // Check that we compiled arithmetic instructions
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("Add"));
    assert!(stats.instruction_counts.contains_key("Sub"));
    assert!(stats.instruction_counts.contains_key("Mul"));

    println!("✅ Arithmetic test passed!");
    println!(
        "   Add: {}, Sub: {}, Mul: {}",
        stats.instruction_counts["Add"],
        stats.instruction_counts["Sub"],
        stats.instruction_counts["Mul"]
    );
}

#[test]
fn test_select_instruction() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("select_test");
    let builder = context.create_builder();

    // Create function: i32 @test_select(i1 %cond, i32 %a, i32 %b)
    let i32_type = context.i32_type();
    let i1_type = context.bool_type();
    let fn_type = i32_type.fn_type(&[i1_type.into(), i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_select", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let cond = function.get_nth_param(0).unwrap().into_int_value();
    let a = function.get_nth_param(1).unwrap().into_int_value();
    let b = function.get_nth_param(2).unwrap().into_int_value();

    // Build select: result = cond ? a : b
    let result = builder.build_select(cond, a, b, "select_result").unwrap();

    builder.build_return(Some(&result)).unwrap();

    // Verify module
    assert!(module.verify().is_ok());

    // Print for debugging
    module.print_to_stderr();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Compile the module
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("test_select").unwrap();

    // Check that we compiled the select instruction
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("Select"));

    println!("✅ Select instruction compiled successfully!");
    println!("   Select: {}", stats.instruction_counts["Select"]);
}

#[test]
fn test_select_with_comparison() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("select_cmp_test");
    let builder = context.create_builder();

    // Create function: i32 @max(i32 %x, i32 %y)
    let i32_type = context.i32_type();
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("max", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let x = function.get_nth_param(0).unwrap().into_int_value();
    let y = function.get_nth_param(1).unwrap().into_int_value();

    // Compare x > y
    let cmp = builder
        .build_int_compare(IntPredicate::SGT, x, y, "cmp")
        .unwrap();

    // Select max(x, y) using: cmp ? x : y
    let max_val = builder.build_select(cmp, x, y, "max").unwrap();

    builder.build_return(Some(&max_val)).unwrap();

    // Verify module
    assert!(module.verify().is_ok());

    // Print for debugging
    module.print_to_stderr();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Compile the module
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("max").unwrap();

    // Check that we compiled both icmp and select instructions
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("ICmp"));
    assert!(stats.instruction_counts.contains_key("Select"));

    println!("✅ Select with comparison compiled successfully!");
    println!(
        "   ICmp: {}, Select: {}",
        stats.instruction_counts["ICmp"], stats.instruction_counts["Select"]
    );
}

#[test]
fn test_floating_point_add() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("test_fadd");
    let builder = context.create_builder();

    // Create function: float add_floats(float a, float b)
    let f32_type = context.f32_type();
    let fn_type = f32_type.fn_type(&[f32_type.into(), f32_type.into()], false);
    let function = module.add_function("add_floats", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let a = function.get_nth_param(0).unwrap().into_float_value();
    let b = function.get_nth_param(1).unwrap().into_float_value();

    let sum = builder.build_float_add(a, b, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();

    // Verify module
    assert!(module.verify().is_ok());

    // Print for debugging
    module.print_to_stderr();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Compile the module
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("add_floats").unwrap();

    // Check that we compiled FAdd instruction
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("FAdd"));

    println!("✅ Float add compiled successfully!");
    println!("   FAdd: {}", stats.instruction_counts["FAdd"]);
}

#[test]
fn test_floating_point_arithmetic() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("test_fp_arithmetic");
    let builder = context.create_builder();

    // Create function: double compute(double x, double y)
    let f64_type = context.f64_type();
    let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
    let function = module.add_function("compute", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let x = function.get_nth_param(0).unwrap().into_float_value();
    let y = function.get_nth_param(1).unwrap().into_float_value();

    // result = (x + y) * (x - y) / 2.0
    let sum = builder.build_float_add(x, y, "sum").unwrap();
    let diff = builder.build_float_sub(x, y, "diff").unwrap();
    let product = builder.build_float_mul(sum, diff, "product").unwrap();
    let two = f64_type.const_float(2.0);
    let result = builder.build_float_div(product, two, "result").unwrap();

    builder.build_return(Some(&result)).unwrap();

    // Verify module
    assert!(module.verify().is_ok());

    // Print for debugging
    module.print_to_stderr();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Compile the module
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("compute").unwrap();

    // Check that we compiled all floating point instructions
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("FAdd"));
    assert!(stats.instruction_counts.contains_key("FSub"));
    assert!(stats.instruction_counts.contains_key("FMul"));
    assert!(stats.instruction_counts.contains_key("FDiv"));

    println!("✅ Float arithmetic compiled successfully!");
    println!(
        "   FAdd: {}, FSub: {}, FMul: {}, FDiv: {}",
        stats.instruction_counts["FAdd"],
        stats.instruction_counts["FSub"],
        stats.instruction_counts["FMul"],
        stats.instruction_counts["FDiv"]
    );
}

#[test]
fn test_floating_point_comparison() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("test_fcmp");
    let builder = context.create_builder();

    // Create function: bool is_greater(float a, float b)
    let f32_type = context.f32_type();
    let i1_type = context.bool_type();
    let fn_type = i1_type.fn_type(&[f32_type.into(), f32_type.into()], false);
    let function = module.add_function("is_greater", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let a_param = function.get_nth_param(0).unwrap();
    let b_param = function.get_nth_param(1).unwrap();

    // Debug print the types
    println!("a_param type: {:?}", a_param.get_type());
    println!("b_param type: {:?}", b_param.get_type());

    // Ensure they are float values
    let a = a_param.into_float_value();
    let b = b_param.into_float_value();

    use inkwell::FloatPredicate;
    let cmp = builder
        .build_float_compare(FloatPredicate::OGT, a, b, "cmp")
        .unwrap();
    builder.build_return(Some(&cmp)).unwrap();

    // Verify module
    assert!(module.verify().is_ok());

    // Print for debugging
    module.print_to_stderr();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Compile the module
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("is_greater").unwrap();

    // Check that we compiled FCmp instruction
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("FCmp"));

    println!("✅ Float comparison compiled successfully!");
    println!("   FCmp: {}", stats.instruction_counts["FCmp"]);
}

#[test]
fn test_mixed_int_float_operations() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("test_mixed");
    let builder = context.create_builder();

    // Create function: float scale_by_int(float value, int scale)
    let f32_type = context.f32_type();
    let i32_type = context.i32_type();
    let fn_type = f32_type.fn_type(&[f32_type.into(), i32_type.into()], false);
    let function = module.add_function("scale_by_int", fn_type, None);

    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let value = function.get_nth_param(0).unwrap().into_float_value();
    let scale_int = function.get_nth_param(1).unwrap().into_int_value();

    // Convert int to float
    let scale_float = builder
        .build_signed_int_to_float(scale_int, f32_type, "scale_float")
        .unwrap();

    // Multiply
    let result = builder
        .build_float_mul(value, scale_float, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    // Verify module
    assert!(module.verify().is_ok());

    // Print for debugging
    module.print_to_stderr();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Compile the module
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("scale_by_int").unwrap();

    // Check that we compiled the conversion and multiplication
    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("SIToFP"));
    assert!(stats.instruction_counts.contains_key("FMul"));

    println!("✅ Mixed int/float operations compiled successfully!");
    println!(
        "   SIToFP: {}, FMul: {}",
        stats.instruction_counts["SIToFP"], stats.instruction_counts["FMul"]
    );
}
