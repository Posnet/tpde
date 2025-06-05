//! Simple test for just return instruction debugging.

use tpde_core::complete_compiler::CompleteCompiler;
use tpde_llvm::enhanced_adaptor::EnhancedLlvmAdaptor;
use inkwell::{context::Context, module::Module};

fn create_simple_return_function(context: &Context) -> Module {
    let module = context.create_module("simple_return");
    let i32_type = context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("test_return", fn_type, None);
    
    let basic_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    
    builder.position_at_end(basic_block);
    let return_value = i32_type.const_int(42, false);
    builder.build_return(Some(&return_value)).unwrap();
    
    module
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Simple Return Instruction Test\n");
    
    let context = Context::create();
    let module = create_simple_return_function(&context);
    let adaptor = EnhancedLlvmAdaptor::new(&module);
    let mut compiler = CompleteCompiler::new(adaptor)?;
    
    println!("Testing single return instruction: ret i32 42");
    
    let funcs: Vec<_> = compiler.adaptor().funcs().collect();
    if let Some(func) = funcs.first() {
        match compiler.compile_function(*func) {
            Ok(()) => println!("✅ Simple return compiled successfully"),
            Err(e) => println!("❌ Error: {}", e),
        }
    }
    
    Ok(())
}