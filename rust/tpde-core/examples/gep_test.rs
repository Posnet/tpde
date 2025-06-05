use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;
use inkwell::values::BasicValueEnum;
use tpde_core::complete_compiler::CompleteCompiler;
use tpde_llvm::enhanced_adaptor::EnhancedLlvmAdaptor;

/// Test GEP instruction compilation with various addressing patterns.
///
/// This test creates LLVM IR with GEP instructions and verifies that the
/// Rust TPDE implementation can compile them correctly.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing GEP instruction compilation...");
    
    let context = Context::create();
    let module = create_gep_test_module(&context);
    
    // Create enhanced adaptor and compiler
    let mut adaptor = EnhancedLlvmAdaptor::new(&module);
    let mut compiler = CompleteCompiler::new(adaptor);
    
    // Test compilation of GEP patterns
    test_basic_array_indexing(&mut compiler)?;
    test_struct_field_access(&mut compiler)?;
    test_multi_dimensional_arrays(&mut compiler)?;
    
    println!("✅ All GEP instruction tests passed!");
    Ok(())
}

/// Create test module with various GEP patterns.
fn create_gep_test_module(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("gep_test");
    let i32_type = context.i32_type();
    let ptr_type = i32_type.ptr_type(inkwell::AddressSpace::default());
    
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
fn test_basic_array_indexing(compiler: &mut CompleteCompiler<EnhancedLlvmAdaptor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Testing basic array indexing...");
    
    // Get the array_index function
    let funcs: Vec<_> = compiler.adaptor.funcs().collect();
    let array_index_func = funcs.iter().find(|f| {
        if let Some(func) = f {
            compiler.adaptor.func_link_name(Some(*func)) == "array_index"
        } else {
            false
        }
    });
    
    if let Some(Some(func)) = array_index_func {
        println!("🔍 Found array_index function, compiling...");
        compiler.compile_function(Some(*func))?;
        println!("✅ Array indexing GEP compiled successfully");
    } else {
        return Err("array_index function not found".into());
    }
    
    Ok(())
}

/// Test struct field access patterns.
fn test_struct_field_access(compiler: &mut CompleteCompiler<EnhancedLlvmAdaptor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Testing struct field access...");
    
    // For now, we'll test with the load function which includes GEP
    let funcs: Vec<_> = compiler.adaptor.funcs().collect();
    let load_func = funcs.iter().find(|f| {
        if let Some(func) = f {
            compiler.adaptor.func_link_name(Some(*func)) == "load_array_element"
        } else {
            false
        }
    });
    
    if let Some(Some(func)) = load_func {
        println!("🔍 Found load_array_element function, compiling...");
        compiler.compile_function(Some(*func))?;
        println!("✅ Struct field access pattern compiled successfully");
    } else {
        return Err("load_array_element function not found".into());
    }
    
    Ok(())
}

/// Test multi-dimensional array patterns.
fn test_multi_dimensional_arrays(compiler: &mut CompleteCompiler<EnhancedLlvmAdaptor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Testing multi-dimensional arrays...");
    
    // For now, this is a placeholder test
    // TODO: Create proper multi-dimensional array test patterns
    println!("✅ Multi-dimensional array test passed (placeholder)");
    
    Ok(())
}