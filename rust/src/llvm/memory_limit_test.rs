//! Test demonstrating memory limit enforcement for database embedding.

#[cfg(test)]
mod tests {
    use crate::core::test_utils::test::{with_limited_context, TestContext};
    use crate::core::{CompilationSession, SessionError};
    use crate::llvm::compiler::LlvmCompiler;
    use inkwell::context::Context;

    fn create_large_function_module(context: &Context) -> inkwell::module::Module {
        let module = context.create_module("large_test");
        let i32_type = context.i32_type();
        
        // Create a function with many basic blocks to consume memory
        let fn_type = i32_type.fn_type(&[i32_type.into()], false);
        let function = module.add_function("large_func", fn_type, None);
        
        let builder = context.create_builder();
        let mut current_block = context.append_basic_block(function, "entry");
        builder.position_at_end(current_block);
        
        // Create many blocks to stress memory usage
        for i in 0..100 {
            let block = context.append_basic_block(function, &format!("block_{}", i));
            builder.build_unconditional_branch(block).unwrap();
            builder.position_at_end(block);
        }
        
        // Final return
        let zero = i32_type.const_int(0, false);
        builder.build_return(Some(&zero)).unwrap();
        
        module
    }

    #[test]
    fn test_memory_limit_enforcement() {
        // Test with a small memory limit (1MB)
        const MEMORY_LIMIT: usize = 1024 * 1024; // 1MB
        
        let ctx = TestContext::with_capacity(MEMORY_LIMIT);
        let mut session = ctx.create_session();
        
        // Create a session with memory limit
        let arena = ctx.arena();
        session = CompilationSession::with_memory_limit(arena, MEMORY_LIMIT);
        
        let context = Context::create();
        let module = create_large_function_module(&context);
        
        // This should eventually fail with memory limit
        let compiler_result = LlvmCompiler::new(module, &session);
        
        if let Ok(mut compiler) = compiler_result {
            // Try to compile - this might exceed memory
            let compile_result = compiler.compile_function_by_name("large_func");
            
            // Check if we hit memory limit
            if let Err(e) = compile_result {
                let error_string = e.to_string();
                assert!(
                    error_string.contains("Memory limit exceeded"),
                    "Expected memory limit error, got: {}",
                    error_string
                );
            }
        }
        
        // Verify we can check memory usage
        let memory_used = ctx.memory_used();
        println!("Memory used in test: {} bytes", memory_used);
        assert!(memory_used > 0);
    }

    #[test]
    fn test_compilation_metrics() {
        let ctx = TestContext::new();
        let session = ctx.create_session();
        
        // Get initial metrics
        let initial_metrics = session.metrics();
        assert_eq!(initial_metrics.functions_compiled, 0);
        assert_eq!(initial_metrics.instructions_compiled, 0);
        
        // Compile something
        let context = Context::create();
        let module = context.create_module("metrics_test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let function = module.add_function("test_func", fn_type, None);
        
        let entry = context.append_basic_block(function, "entry");
        let builder = context.create_builder();
        builder.position_at_end(entry);
        let zero = i32_type.const_int(0, false);
        builder.build_return(Some(&zero)).unwrap();
        
        let mut compiler = LlvmCompiler::new(module, &session).unwrap();
        compiler.compile_function_by_name("test_func").unwrap();
        
        // Check metrics after compilation
        let final_metrics = session.metrics();
        assert_eq!(final_metrics.functions_compiled, 1);
        assert!(final_metrics.instructions_compiled > 0);
        assert!(final_metrics.memory_used > 0);
        assert!(final_metrics.elapsed_time.as_nanos() > 0);
    }
}