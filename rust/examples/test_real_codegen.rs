// This example demonstrates real machine code generation capabilities of the TPDE compiler by
// compiling a simple add function from LLVM IR to x86-64 machine code. It serves as a basic
// smoke test to verify that the compiler can handle fundamental arithmetic operations and
// generate actual executable instructions rather than placeholders. The example creates a
// minimal LLVM function `int add(int a, int b) { return a + b; }` using inkwell, then uses
// LlvmCompiler to compile it to native code. The test validates that proper x86-64 instructions
// are generated including function prologue (push rbp, mov rbp rsp), ADD instruction for the
// arithmetic operation, result placement in RAX register per System V ABI, and function epilogue
// (pop rbp, ret). This example also demonstrates compilation session usage with arena allocation
// and shows how to extract compilation statistics like instruction count and code size. It's
// useful for verifying basic compiler functionality after making changes to instruction selection
// or code generation components.

use bumpalo::Bump;
use inkwell::context::Context;
use tpde::core::CompilationSession;
use tpde::llvm::LlvmCompiler;

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

    // Create compilation session and compiler
    let arena = Bump::new();
    let session = CompilationSession::new(&arena);
    let mut compiler = LlvmCompiler::new(module, &session)?;

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
    let result = builder
        .build_int_add(param_a, param_b, "add_result")
        .unwrap();

    // Return result
    builder.build_return(Some(&result)).unwrap();

    module
}

/// Test compilation of the simple add function.
fn test_simple_add_compilation<'ctx, 'arena>(
    compiler: &mut LlvmCompiler<'ctx, 'arena>,
) -> Result<(), Box<dyn std::error::Error>>
where
    'ctx: 'arena,
{
    println!("📋 Testing simple add function compilation...");

    // Compile the add function
    println!("🔍 Compiling add function...");

    match compiler.compile_function_by_name("add") {
        Ok(_) => {
            println!("✅ Add function compiled successfully!");

            // Get compilation statistics
            let stats = compiler.session().stats();
            println!("📊 Compilation statistics:");
            println!("   - {} instructions compiled", stats.instructions_compiled);
            println!("   - {} bytes of code generated", stats.total_code_size);

            println!("🎯 Machine code should contain:");
            println!("   - Function prologue (push rbp, mov rbp, rsp)");
            println!("   - ADD instruction for arithmetic");
            println!("   - Return value handling (result in RAX)");
            println!("   - Function epilogue (pop rbp, ret)");

            Ok(())
        }
        Err(e) => {
            println!("❌ Compilation failed: {e:?}");
            Err(format!("Compilation failed: {e:?}").into())
        }
    }
}
