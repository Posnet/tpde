// build.rs

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// LLVM API imports
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{InitializationConfig, Target, TargetTriple, CodeModel, RelocMode, FileType};
use inkwell::OptimizationLevel;

fn main() {
    println!("cargo:rerun-if-changed=src/templates/");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let target = env::var("TARGET").unwrap();
    
    // Check if we should extract templates
    if env::var("TPDE_EXTRACT_TEMPLATES").is_ok() {
        eprintln!("Extracting instruction patterns from templates...");
        extract_and_generate(&out_dir, &target);
    } else {
        // Use pre-generated or placeholder encoders
        generate_placeholder_encoders(&out_dir);
    }
}

fn extract_and_generate(out_dir: &str, _target: &str) {
    // Initialize LLVM targets
    Target::initialize_x86(&InitializationConfig::default());
    Target::initialize_aarch64(&InitializationConfig::default());
    
    // Step 1: Compile templates to LLVM Module
    let context = Context::create();
    let module = compile_templates_to_module(&context, out_dir, _target);
    
    // Step 2: Optimize the module to get clean patterns
    optimize_module(&module);
    
    // Step 3: Generate assembly and extract patterns
    // Since we can't access MachineInstr from Rust, we'll use a hybrid approach
    let patterns = extract_patterns_from_module(&context, &module, out_dir);
    
    // Step 4: Generate encoder implementations
    generate_encoder_impls(&patterns, out_dir);
}

fn compile_templates_to_module<'ctx>(context: &'ctx Context, out_dir: &str, _target: &str) -> Module<'ctx> {
    // For now, we'll create the module programmatically
    // In the future, we could compile templates and load the IR
    let module = context.create_module("templates");
    let builder = context.create_builder();
    
    // Set target triple
    let triple = TargetTriple::create("x86_64-unknown-linux-gnu");
    module.set_triple(&triple);
    
    // Create template functions programmatically
    // add_i32(i32, i32) -> i32
    {
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("add_i32", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        builder.position_at_end(basic_block);
        let param0 = function.get_nth_param(0).unwrap().into_int_value();
        let param1 = function.get_nth_param(1).unwrap().into_int_value();
        let result = builder.build_int_add(param0, param1, "add").unwrap();
        builder.build_return(Some(&result)).unwrap();
    }
    
    // sub_i32(i32, i32) -> i32
    {
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("sub_i32", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        builder.position_at_end(basic_block);
        let param0 = function.get_nth_param(0).unwrap().into_int_value();
        let param1 = function.get_nth_param(1).unwrap().into_int_value();
        let result = builder.build_int_sub(param0, param1, "sub").unwrap();
        builder.build_return(Some(&result)).unwrap();
    }
    
    // mul_i32(i32, i32) -> i32
    {
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("mul_i32", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        builder.position_at_end(basic_block);
        let param0 = function.get_nth_param(0).unwrap().into_int_value();
        let param1 = function.get_nth_param(1).unwrap().into_int_value();
        let result = builder.build_int_mul(param0, param1, "mul").unwrap();
        builder.build_return(Some(&result)).unwrap();
    }
    
    // add_f32(f32, f32) -> f32
    {
        let f32_type = context.f32_type();
        let fn_type = f32_type.fn_type(&[f32_type.into(), f32_type.into()], false);
        let function = module.add_function("add_f32", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        builder.position_at_end(basic_block);
        let param0 = function.get_nth_param(0).unwrap().into_float_value();
        let param1 = function.get_nth_param(1).unwrap().into_float_value();
        let result = builder.build_float_add(param0, param1, "fadd").unwrap();
        builder.build_return(Some(&result)).unwrap();
    }
    
    // icmp_eq_i32(i32, i32) -> i1
    {
        let i32_type = context.i32_type();
        let i1_type = context.bool_type();
        let fn_type = i1_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("icmp_eq_i32", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        builder.position_at_end(basic_block);
        let param0 = function.get_nth_param(0).unwrap().into_int_value();
        let param1 = function.get_nth_param(1).unwrap().into_int_value();
        let result = builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            param0,
            param1,
            "cmp"
        ).unwrap();
        builder.build_return(Some(&result)).unwrap();
    }
    
    // select_i32(i1, i32, i32) -> i32
    {
        let i32_type = context.i32_type();
        let i1_type = context.bool_type();
        let fn_type = i32_type.fn_type(&[i1_type.into(), i32_type.into(), i32_type.into()], false);
        let function = module.add_function("select_i32", fn_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        
        builder.position_at_end(basic_block);
        let cond = function.get_nth_param(0).unwrap().into_int_value();
        let val_true = function.get_nth_param(1).unwrap().into_int_value();
        let val_false = function.get_nth_param(2).unwrap().into_int_value();
        let result = builder.build_select(cond, val_true, val_false, "select").unwrap();
        builder.build_return(Some(&result)).unwrap();
    }
    
    // Verify the module
    module.verify().expect("Module verification failed");
    
    // Save IR to file for debugging
    let ir_path = Path::new(out_dir).join("templates.ll");
    module.print_to_file(&ir_path).expect("Failed to write LLVM IR");
    eprintln!("Generated LLVM IR at: {:?}", ir_path);
    
    module
}

fn optimize_module(module: &Module) {
    // For now, just verify the module
    // In a real implementation, we would run optimization passes
    module.verify().expect("Module verification failed");
}

fn extract_patterns_from_module<'ctx>(_context: &'ctx Context, module: &Module<'ctx>, out_dir: &str) -> Vec<Pattern> {
    let mut patterns = Vec::new();
    
    // Create target machine for x86-64
    let target_triple = TargetTriple::create("x86_64-unknown-linux-gnu");
    let target = Target::from_triple(&target_triple).expect("Failed to get target");
    let target_machine = target.create_target_machine(
        &target_triple,
        "x86-64",
        "",
        OptimizationLevel::Aggressive,
        RelocMode::Default,
        CodeModel::Default,
    ).expect("Failed to create target machine");
    
    // Generate assembly to file
    let asm_path = Path::new(out_dir).join("templates.s");
    target_machine.write_to_file(
        module,
        FileType::Assembly,
        &asm_path,
    ).expect("Failed to write assembly");
    
    // Also analyze the LLVM IR for each function
    for function in module.get_functions() {
        if function.count_basic_blocks() == 0 {
            continue;
        }
        
        let name = function.get_name().to_str().unwrap().to_string();
        let mut instructions = Vec::new();
        
        // Analyze LLVM IR instructions
        for bb in function.get_basic_blocks() {
            for inst in bb.get_instructions() {
                // Extract instruction patterns from LLVM IR
                let inst_str = format!("{:?}", inst);
                
                // Parse the instruction type
                if inst_str.contains("add ") {
                    instructions.push(Instruction {
                        mnemonic: "add".to_string(),
                        operands: "llvm_pattern".to_string(),
                    });
                } else if inst_str.contains("sub ") {
                    instructions.push(Instruction {
                        mnemonic: "sub".to_string(),
                        operands: "llvm_pattern".to_string(),
                    });
                } else if inst_str.contains("mul ") {
                    instructions.push(Instruction {
                        mnemonic: "mul".to_string(),
                        operands: "llvm_pattern".to_string(),
                    });
                } else if inst_str.contains("fadd ") {
                    instructions.push(Instruction {
                        mnemonic: "fadd".to_string(),
                        operands: "llvm_pattern".to_string(),
                    });
                } else if inst_str.contains("icmp ") {
                    instructions.push(Instruction {
                        mnemonic: "icmp".to_string(),
                        operands: "llvm_pattern".to_string(),
                    });
                } else if inst_str.contains("select ") {
                    instructions.push(Instruction {
                        mnemonic: "select".to_string(),
                        operands: "llvm_pattern".to_string(),
                    });
                }
            }
        }
        
        if !instructions.is_empty() {
            patterns.push(Pattern {
                name,
                instructions,
            });
        }
    }
    
    // Also parse the assembly file for more detailed patterns
    let asm_content = fs::read_to_string(&asm_path).expect("Failed to read assembly");
    parse_assembly_patterns(&asm_content, &mut patterns);
    
    eprintln!("Extracted {} patterns", patterns.len());
    patterns
}

fn parse_assembly_patterns(_asm_content: &str, _patterns: &mut Vec<Pattern>) {
    // This could enhance the patterns with actual x86 instructions
    // For now, we'll just use the LLVM IR patterns
}

fn extract_patterns_from_asm(asm_path: &Path) -> Vec<Pattern> {
    let asm_content = fs::read_to_string(asm_path)
        .expect("Failed to read assembly file");
    
    let mut patterns = Vec::new();
    let mut current_func = None;
    let mut instructions = Vec::new();
    
    for line in asm_content.lines() {
        let line = line.trim();
        
        // Function label
        if line.ends_with(':') && !line.starts_with('.') {
            // Save previous function if any
            if let Some(name) = current_func.take() {
                if !instructions.is_empty() {
                    patterns.push(Pattern {
                        name,
                        instructions: instructions.clone(),
                    });
                    instructions.clear();
                }
            }
            
            current_func = Some(line.trim_end_matches(':').to_string());
        }
        // Instruction line
        else if let Some(_) = &current_func {
            // Skip empty lines, directives, and comments
            if !line.is_empty() && !line.trim().starts_with('.') && !line.trim().starts_with('#') {
                // Simple instruction parsing
                // Handle both tab and space separated instructions
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() {
                    let mnemonic = parts[0].to_string();
                    let operands = if parts.len() > 1 {
                        parts[1..].join(" ")
                    } else {
                        String::new()
                    };
                    
                    // Skip labels and directives
                    if !mnemonic.ends_with(':') && !mnemonic.starts_with('.') {
                        instructions.push(Instruction {
                            mnemonic,
                            operands,
                        });
                    }
                }
                
                // End at ret instruction
                if line.trim().starts_with("ret") {
                    if let Some(name) = current_func.take() {
                        patterns.push(Pattern {
                            name,
                            instructions: instructions.clone(),
                        });
                        instructions.clear();
                    }
                }
            }
        }
    }
    
    eprintln!("Extracted {} patterns", patterns.len());
    for pattern in &patterns {
        eprintln!("  {}: {} instructions", pattern.name, pattern.instructions.len());
        for inst in &pattern.instructions {
            eprintln!("    {} {}", inst.mnemonic, inst.operands);
        }
    }
    patterns
}

#[derive(Debug)]
struct Pattern {
    name: String,
    instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
struct Instruction {
    mnemonic: String,
    operands: String,
}

fn generate_encoder_impls(patterns: &[Pattern], out_dir: &str) {
    let mut code = String::from(
        "// GENERATED CODE - DO NOT EDIT\n\
         // Generated by build.rs from template patterns\n\n\
         use crate::core::{ValuePartRef, AsmReg};\n\
         use crate::x64::encoder::{X64Encoder, EncodingError};\n\n"
    );
    
    // Generate an encoder function for each pattern
    for pattern in patterns {
        // Map template function names to encoder function names
        let encoder_name = format!("encode_{}", pattern.name);
        
        code.push_str(&format!(
            "/// Generated encoder for {}\n\
             /// Instructions: {:?}\n\
             pub fn {}<'arena>(\n    \
                 encoder: &mut X64Encoder<'arena>,\n    \
                 inputs: &[ValuePartRef],\n    \
                 output: AsmReg,\n\
             ) -> Result<(), EncodingError> {{\n",
            pattern.name,
            pattern.instructions.iter().map(|i| &i.mnemonic).collect::<Vec<_>>(),
            encoder_name
        ));
        
        // Generate encoding based on the pattern name and type
        // For now, generate placeholders that compile
        code.push_str(&format!("    // Pattern: {} with {} instructions\n", 
            pattern.name, pattern.instructions.len()));
        
        for inst in &pattern.instructions {
            code.push_str(&format!("    // Instruction: {} {}\n", inst.mnemonic, inst.operands));
        }
        
        code.push_str("    // TODO: Implement template-based encoding\n");
        code.push_str("    let _ = (encoder, inputs, output);\n");
        code.push_str("    todo!(\"Template-based encoder not yet implemented\")\n");
        
        code.push_str("}\n\n");
    }
    
    fs::write(Path::new(out_dir).join("generated_encoders.rs"), code)
        .expect("Failed to write generated encoders");
}

fn generate_placeholder_encoders(out_dir: &str) {
    let code = r#"
// GENERATED CODE - DO NOT EDIT
// Placeholder encoders - run with TPDE_EXTRACT_TEMPLATES=1 to generate real ones
use crate::core::{ValuePartRef, AsmReg};
use crate::x64::encoder::{X64Encoder, EncodingError};

pub fn encode_add_i32<'arena>(
    encoder: &mut X64Encoder<'arena>,
    inputs: &[ValuePartRef],
    output: AsmReg,
) -> Result<(), EncodingError> {
    todo!("Run with TPDE_EXTRACT_TEMPLATES=1 to generate real encoder")
}

pub fn encode_add_f32<'arena>(
    encoder: &mut X64Encoder<'arena>,
    inputs: &[ValuePartRef],
    output: AsmReg,
) -> Result<(), EncodingError> {
    todo!("Run with TPDE_EXTRACT_TEMPLATES=1 to generate real encoder")
}

// Add more placeholder functions as needed
"#;
    
    fs::write(Path::new(out_dir).join("generated_encoders.rs"), code)
        .expect("Failed to write placeholder encoders");
}// force rebuild
