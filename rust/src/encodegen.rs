// This module provides functionality for generating Rust instruction encoding snippets from
// LLVM IR patterns, formerly part of the tpde-encodegen crate. It analyzes LLVM IR functions
// whose names start with "pattern_" and generates corresponding Rust functions that emit
// machine instructions. The module parses LLVM IR text/modules using inkwell, extracts
// pattern functions, analyzes their instruction sequences (add, sub, mul, ret, store, load,
// call, br), and produces Rust code stubs with encoding hints. This is used to transform
// high-level C functions compiled to LLVM IR into low-level instruction emission routines
// for the TPDE compiler backend. The generated code provides templates for x86/ARM instruction
// encoding based on the LLVM IR structure.

//! Encoding snippet generator module.
//!
//! This module contains the functionality that was previously in the
//! tpde-encodegen crate. It transforms short high level functions 
//! (typically written in C and compiled to LLVM IR) into Rust routines 
//! that emit the matching machine instructions.

use inkwell::{
    context::Context,
    memory_buffer::MemoryBuffer,
    module::Module,
};

/// Parse a text LLVM IR module into an [`inkwell::Module`].
pub fn parse_module<'ctx>(context: &'ctx Context, ir: &str) -> Result<Module<'ctx>, String> {
    let buffer = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), "ir");
    context
        .create_module_from_ir(buffer)
        .map_err(|e| e.to_string())
}

/// Generate Rust source snippets for functions starting with `pattern_`.
///
/// Every matching function name is turned into a small Rust stub that
/// analyzes the LLVM IR structure and generates appropriate encoding functions.
pub fn generate_tokens(module: &Module) -> Vec<String> {
    module
        .get_functions()
        .filter_map(|f| {
            let name = f.get_name().to_str().ok()?;
            if let Some(rest) = name.strip_prefix("pattern_") {
                let body = analyze_function(&f);
                Some(format!("pub fn {}(asm: &mut dyn Assembler) {{\n{}\n}}", rest, body))
            } else {
                None
            }
        })
        .collect()
}

/// Analyze a function and generate appropriate encoding code.
fn analyze_function(func: &inkwell::values::FunctionValue) -> String {
    let mut lines = vec!["    // Generated from LLVM IR pattern".to_string()];
    
    for bb in func.get_basic_blocks() {
        for inst in bb.get_instructions() {
            let analysis = analyze_instruction(&inst);
            if !analysis.is_empty() {
                lines.push(format!("    {}", analysis));
            }
        }
    }
    
    if lines.len() == 1 {
        lines.push("    // Empty pattern - no instructions to encode".to_string());
    }
    
    lines.join("\n")
}

/// Analyze a single instruction and generate encoding suggestions.
fn analyze_instruction(inst: &inkwell::values::InstructionValue) -> String {
    use inkwell::values::InstructionOpcode;
    
    match inst.get_opcode() {
        InstructionOpcode::Add => "// Add instruction - could emit x86 ADD",
        InstructionOpcode::Sub => "// Sub instruction - could emit x86 SUB", 
        InstructionOpcode::Mul => "// Mul instruction - could emit x86 IMUL",
        InstructionOpcode::Return => "// Return instruction - emit x86 RET",
        InstructionOpcode::Store => "// Store instruction - emit x86 MOV to memory",
        InstructionOpcode::Load => "// Load instruction - emit x86 MOV from memory",
        InstructionOpcode::Call => "// Call instruction - emit x86 CALL",
        InstructionOpcode::Br => "// Branch instruction - emit x86 JMP",
        _ => "// Unhandled instruction type",
    }.to_string()
}

/// Convenience helper parsing IR text and returning token strings.
pub fn parse_and_generate<'ctx>(
    context: &'ctx Context,
    ir: &str,
) -> Result<Vec<String>, String> {
    let module = parse_module(context, ir)?;
    Ok(generate_tokens(&module))
}

/// Generate snippet encoder source from the provided LLVM IR `Module`.
///
/// This analyzes pattern functions in the module and generates Rust code
/// for instruction encoding based on the LLVM IR structure.
pub fn generate(module: &Module) -> String {
    let tokens = generate_tokens(module);
    if tokens.is_empty() {
        "// No pattern functions found\n".to_string()
    } else {
        format!("// Generated instruction encoders\n\n{}\n", tokens.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_generate_empty_module() {
        let context = Context::create();
        let module = context.create_module("test");
        let result = generate(&module);
        assert!(result.contains("No pattern functions found"));
    }

    #[test]
    fn test_generate_pattern_function() {
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let _function = module.add_function("pattern_add_i32", fn_type, None);
        
        let tokens = generate_tokens(&module);
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].contains("add_i32"));
        assert!(tokens[0].contains("Generated from LLVM IR pattern"));
    }

    #[test]
    fn test_parse_and_generate_ir() {
        let ir = r#"
            define i32 @pattern_simple_add() {
                %1 = add i32 1, 2
                ret i32 %1
            }
        "#;
        
        let context = Context::create();
        let result = parse_and_generate(&context, ir);
        assert!(result.is_ok());
        
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].contains("simple_add"));
        assert!(tokens[0].contains("Add instruction"));
        assert!(tokens[0].contains("Return instruction"));
    }
}