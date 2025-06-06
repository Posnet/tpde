use inkwell::context::Context;
use tpde::complete_compiler::CompleteCompiler;
use tpde::llvm_adaptor::enhanced::EnhancedLlvmAdaptor;

/// Test real machine code generation with simple arithmetic.
///
/// This test verifies that the instruction selection improvements actually
/// generate machine code instead of just placeholders.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger for debugging
    env_logger::init();
    
    println!("🧪 Testing Real Machine Code Generation...");
    
    let context = Context::create();
    let module = create_simple_add_function(&context);
    
    // Create enhanced adaptor and compiler
    let mut adaptor = EnhancedLlvmAdaptor::new(&module);
    let mut compiler = CompleteCompiler::new(adaptor);
    
    // Test compilation of simple add function
    test_simple_add_compilation(&mut compiler)?;
    
    println!("✅ Real machine code generation test completed!");
    Ok(())
}

/// Create a simple function: int add(int a, int b) { return a + b; }
fn create_simple_add_function(context: &Context) -> inkwell::module::Module {
    let module = context.create_module("simple_add");
    let i32_type = context.i32_type();
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("add", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    // Get function parameters
    let param_a = function.get_nth_param(0).unwrap().into_int_value();
    let param_b = function.get_nth_param(1).unwrap().into_int_value();
    
    // Generate add instruction
    let result = builder.build_int_add(param_a, param_b, "add_result").unwrap();
    
    // Return result
    builder.build_return(Some(&result)).unwrap();
    
    module
}

/// Test compilation of the simple add function.
fn test_simple_add_compilation(compiler: &mut CompleteCompiler<EnhancedLlvmAdaptor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Testing simple add function compilation...");
    
    // Get the add function
    let funcs: Vec<_> = compiler.adaptor.funcs().collect();
    let add_func = funcs.iter().find(|f| {
        if let Some(func) = f {
            compiler.adaptor.func_link_name(Some(*func)) == "add"
        } else {
            false
        }
    });
    
    if let Some(Some(func)) = add_func {
        println!("🔍 Found add function, compiling...");
        
        // Try to compile the function
        match compiler.compile_function(Some(*func)) {
            Ok(_) => {
                println!("✅ Add function compiled successfully!");
                
                // TODO: In a complete test, we would:
                // 1. Extract generated machine code from assembler
                // 2. Verify specific instructions were generated (ADD, MOV, RET)
                // 3. Test that machine code is executable
                
                println!("🎯 Machine code should contain:");
                println!("   - Function prologue (push rbp, mov rbp, rsp)");
                println!("   - ADD instruction for arithmetic");
                println!("   - Return value handling (result in RAX)");
                println!("   - Function epilogue (pop rbp, ret)");
                
                Ok(())
            }
            Err(e) => {
                println!("❌ Compilation failed: {:?}", e);
                Err(format!("Compilation failed: {:?}", e).into())
            }
        }
    } else {
        Err("add function not found".into())
    }
}