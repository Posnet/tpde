//! Test advanced calling convention features including byval, sret, varargs.

use bumpalo::Bump;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use tpde::{core::CompilationSession, llvm::LlvmCompiler};

/// Create a test module with byval parameter.
fn create_byval_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("byval_test");
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let struct_type = context.struct_type(&[i32_type.into(), i64_type.into()], false);
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // Function: void process_struct(%struct* byval)
    let fn_type = context.void_type().fn_type(&[ptr_type.into()], false);
    let function = module.add_function("process_struct", fn_type, None);

    // Add byval attribute to first parameter
    let byval_kind_id = Attribute::get_named_enum_kind_id("byval");
    let byval_attr = context.create_enum_attribute(byval_kind_id, 0);
    function.add_attribute(AttributeLoc::Param(0), byval_attr);

    // Create a simple function body
    let entry = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);
    builder.build_return(None).unwrap();

    // Create a caller function
    let caller_fn_type = context.void_type().fn_type(&[], false);
    let caller = module.add_function("caller", caller_fn_type, None);
    let caller_entry = context.append_basic_block(caller, "entry");

    builder.position_at_end(caller_entry);

    // Allocate a struct on stack and call with byval
    let stack_struct = builder.build_alloca(struct_type, "tmp_struct").unwrap();

    // Initialize the struct
    let gep0 = builder
        .build_struct_gep(struct_type, stack_struct, 0, "field0")
        .unwrap();
    builder
        .build_store(gep0, i32_type.const_int(42, false))
        .unwrap();

    let gep1 = builder
        .build_struct_gep(struct_type, stack_struct, 1, "field1")
        .unwrap();
    builder
        .build_store(gep1, i64_type.const_int(100, false))
        .unwrap();

    // Call with byval
    let call_site = builder
        .build_call(function, &[stack_struct.into()], "call")
        .unwrap();

    // Add byval attribute to the call argument
    call_site.add_attribute(AttributeLoc::Param(0), byval_attr);

    builder.build_return(None).unwrap();

    // Caller that invokes make_pair
    module
}

/// Create a test module with sret (struct return).
fn create_sret_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("sret_test");
    let i32_type = context.i32_type();
    let struct_type = context.struct_type(&[i32_type.into(), i32_type.into()], false);
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // Function: void make_pair(%struct* sret, i32, i32)
    let fn_type = context
        .void_type()
        .fn_type(&[ptr_type.into(), i32_type.into(), i32_type.into()], false);
    let function = module.add_function("make_pair", fn_type, None);

    // Add sret attribute to first parameter
    let sret_kind_id = Attribute::get_named_enum_kind_id("sret");
    let sret_attr = context.create_enum_attribute(sret_kind_id, 0);
    function.add_attribute(AttributeLoc::Param(0), sret_attr);

    // Function body: store values into sret parameter
    let entry = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);

    let sret_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let val1 = function.get_nth_param(1).unwrap().into_int_value();
    let val2 = function.get_nth_param(2).unwrap().into_int_value();

    let gep0 = builder
        .build_struct_gep(struct_type, sret_ptr, 0, "field0")
        .unwrap();
    builder.build_store(gep0, val1).unwrap();

    let gep1 = builder
        .build_struct_gep(struct_type, sret_ptr, 1, "field1")
        .unwrap();
    builder.build_store(gep1, val2).unwrap();

    builder.build_return(None).unwrap();

    // Caller that invokes make_pair
    let caller_ty = context.void_type().fn_type(&[], false);
    let caller = module.add_function("call_make_pair", caller_ty, None);
    let caller_entry = context.append_basic_block(caller, "entry");
    builder.position_at_end(caller_entry);

    let tmp = builder.build_alloca(struct_type, "tmp_pair").unwrap();
    let v1 = i32_type.const_int(1, false);
    let v2 = i32_type.const_int(2, false);
    let call = builder
        .build_call(function, &[tmp.into(), v1.into(), v2.into()], "call")
        .unwrap();
    call.add_attribute(AttributeLoc::Param(0), sret_attr);
    builder.build_return(None).unwrap();

    module
}

/// Create a test module with varargs function.
fn create_varargs_test(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("varargs_test");
    let i32_type = context.i32_type();

    // Function: i32 sum_ints(i32 count, ...)
    let fn_type = i32_type.fn_type(&[i32_type.into()], true); // true = varargs
    let function = module.add_function("sum_ints", fn_type, None);

    // Simple implementation: just return the count for now
    let entry = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);

    let count = function.get_nth_param(0).unwrap().into_int_value();
    builder.build_return(Some(&count)).unwrap();

    // Create a caller
    let caller_fn_type = i32_type.fn_type(&[], false);
    let caller = module.add_function("test_varargs", caller_fn_type, None);
    let caller_entry = context.append_basic_block(caller, "entry");

    builder.position_at_end(caller_entry);

    // Call with 3 varargs
    let count = i32_type.const_int(3, false);
    let arg1 = i32_type.const_int(10, false);
    let arg2 = i32_type.const_int(20, false);
    let arg3 = i32_type.const_int(30, false);

    let result = builder
        .build_call(
            function,
            &[count.into(), arg1.into(), arg2.into(), arg3.into()],
            "varargs_call",
        )
        .unwrap();

    builder
        .build_return(Some(&result.try_as_basic_value().left().unwrap()))
        .unwrap();

    module
}

#[test]
fn test_byval_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_byval_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the caller function which uses byval
    compiler.compile_function_by_name("caller").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("caller"));

    let stats = compiler.session().stats();
    assert_eq!(stats.functions_compiled, 1);
    assert!(stats.instruction_counts.contains_key("Call"));

    println!("✅ byval test passed!");
}

#[test]
fn test_sret_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_sret_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the caller which triggers the sret call
    compiler.compile_function_by_name("call_make_pair").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("call_make_pair"));

    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("Call"));

    println!("✅ sret test passed!");
}

#[test]
fn test_varargs_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = create_varargs_test(&context);

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the caller which uses varargs
    compiler.compile_function_by_name("test_varargs").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("test_varargs"));

    let stats = compiler.session().stats();
    assert!(stats.instruction_counts.contains_key("Call"));

    println!("✅ varargs test passed!");
}

#[test]
fn test_zext_sext_attributes() {
    let _ = env_logger::builder().is_test(true).try_init();

    let context = Context::create();
    let module = context.create_module("ext_test");

    // Create a function that takes parameters with zext/sext
    let i8_type = context.i8_type();
    let i32_type = context.i32_type();

    // Function: i32 process(i8 zeroext, i8 signext)
    let fn_type = i32_type.fn_type(&[i8_type.into(), i8_type.into()], false);
    let function = module.add_function("process", fn_type, None);

    // Add attributes
    let zext_kind_id = Attribute::get_named_enum_kind_id("zeroext");
    let zext_attr = context.create_enum_attribute(zext_kind_id, 0);
    function.add_attribute(AttributeLoc::Param(0), zext_attr);

    let sext_kind_id = Attribute::get_named_enum_kind_id("signext");
    let sext_attr = context.create_enum_attribute(sext_kind_id, 0);
    function.add_attribute(AttributeLoc::Param(1), sext_attr);

    // Function body
    let entry = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);
    builder
        .build_return(Some(&i32_type.const_int(0, false)))
        .unwrap();

    // Create arena and session
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);

    // Create compiler
    let mut compiler = LlvmCompiler::new(module, &session).unwrap();

    // Compile the function
    compiler.compile_function_by_name("process").unwrap();

    // Verify function was compiled
    let compiled_functions = compiler.compiled_functions();
    assert!(compiled_functions.contains_key("process"));

    println!("✅ zext/sext test passed!");
}
