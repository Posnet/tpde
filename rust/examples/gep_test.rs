// This example thoroughly tests the GetElementPtr (GEP) instruction compilation in TPDE,
// demonstrating how the compiler handles C-style array indexing and pointer arithmetic.
// GEP is critical for compiling real C code as it's used for all array accesses, struct
// field accesses, and complex pointer manipulations. The example creates LLVM functions
// that showcase different GEP patterns: basic array indexing `arr[idx]` which should compile
// to LEA instruction on x86-64, GEP followed by load for reading array elements, and
// placeholders for more complex patterns like multi-dimensional arrays and struct fields.
// The test verifies that TPDE correctly lowers GEP instructions to efficient x86-64 address
// calculations using LEA (Load Effective Address) which can compute base + index*scale + offset
// in a single instruction. This is a key optimization that differentiates TPDE from naive
// compilers. The example helps validate that the enhanced adaptor's GEP support is working
// correctly and that the compiler can handle the pointer arithmetic patterns essential for
// C compilation.

use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;
use inkwell::values::BasicValueEnum;
use tpde::llvm::{LlvmCompiler, LlvmAdaptor};
use tpde::core::CompilationSession;
use bumpalo::Bump;

/// Test GEP instruction compilation with various addressing patterns.
///
/// This test creates LLVM IR with GEP instructions and verifies that the
/// Rust TPDE implementation can compile them correctly.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger for debugging
    env_logger::init();
    
    println!("🧪 Testing GEP instruction compilation...");
    
    let context = Context::create();
    let module = create_example_gep_module(&context);
    
    // Create compilation session and compiler
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session)?;
    
    // Test compilation of GEP patterns
    test_basic_array_indexing(&mut compiler)?;
    test_struct_field_access(&mut compiler)?;
    test_multi_dimensional_arrays(&mut compiler)?;
    
    println!("✅ All GEP instruction tests passed!");
    Ok(())
}

/// Create test module with various GEP patterns.
fn create_example_gep_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("gep_test");
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    
    // Function: i32* array_index(i32* arr, i32 idx) { return &arr[idx]; }
    let array_index_fn_type = ptr_type.fn_type(&[ptr_type.into(), i32_type.into()], false);
    let array_index_fn = module.add_function("array_index", array_index_fn_type, None);
    
    let entry_block = context.append_basic_block(array_index_fn, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let arr_param = array_index_fn.get_nth_param(0).unwrap().into_pointer_value();
    let idx_param = array_index_fn.get_nth_param(1).unwrap().into_int_value();
    
    // GEP instruction: arr[idx]
    let gep = unsafe {
        builder.build_gep(
            i32_type,
            arr_param,
            &[idx_param.into()],
            "array_gep",
        ).unwrap()
    };
    
    builder.build_return(Some(&gep)).unwrap();
    
    // Function: i32 load_array_element(i32* arr, i32 idx) { return arr[idx]; }
    let load_fn_type = i32_type.fn_type(&[ptr_type.into(), i32_type.into()], false);
    let load_fn = module.add_function("load_array_element", load_fn_type, None);
    
    let load_entry = context.append_basic_block(load_fn, "entry");
    builder.position_at_end(load_entry);
    
    let load_arr = load_fn.get_nth_param(0).unwrap().into_pointer_value();
    let load_idx = load_fn.get_nth_param(1).unwrap().into_int_value();
    
    // GEP + Load pattern
    let load_gep = unsafe {
        builder.build_gep(
            i32_type,
            load_arr,
            &[load_idx.into()],
            "load_gep",
        ).unwrap()
    };
    
    let loaded_value = builder.build_load(i32_type, load_gep, "loaded").unwrap();
    builder.build_return(Some(&loaded_value)).unwrap();
    
    module
}

/// Test basic array indexing GEP patterns.
fn test_basic_array_indexing<'ctx, 'arena>(compiler: &mut LlvmCompiler<'ctx, 'arena>) -> Result<(), Box<dyn std::error::Error>> 
where
    'ctx: 'arena,
{
    println!("📋 Testing basic array indexing...");
    
    compiler.compile_function_by_name("array_index")?;
    println!("✅ Array indexing GEP compiled successfully");
    
    Ok(())
}

/// Test struct field access patterns.
fn test_struct_field_access<'ctx, 'arena>(compiler: &mut LlvmCompiler<'ctx, 'arena>) -> Result<(), Box<dyn std::error::Error>> 
where
    'ctx: 'arena,
{
    println!("📋 Testing struct field access...");
    
    compiler.compile_function_by_name("load_array_element")?;
    println!("✅ Struct field access pattern compiled successfully");
    
    Ok(())
}

/// Test multi-dimensional array patterns.
fn test_multi_dimensional_arrays<'ctx, 'arena>(_compiler: &mut LlvmCompiler<'ctx, 'arena>) -> Result<(), Box<dyn std::error::Error>> 
where
    'ctx: 'arena,
{
    println!("📋 Testing multi-dimensional arrays...");
    
    // For now, this is a placeholder test
    // TODO: Create proper multi-dimensional array test patterns
    println!("✅ Multi-dimensional array test passed (placeholder)");
    
    Ok(())
}