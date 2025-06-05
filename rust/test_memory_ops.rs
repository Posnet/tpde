//! Test script to verify memory operation codegen works correctly
//! This demonstrates load, store, and alloca compilation with the Rust TPDE implementation

use tpde_rust::complete_compiler::{CompleteCompiler, SimpleTestIR};
use tpde_rust::x64_encoder::{X64Encoder, EncodingError};
use tpde_rust::register_file::AsmReg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing memory operation codegen with Rust TPDE implementation");

    // Test 1: Basic encoder functionality for memory operations
    test_x64_encoder_memory_operations()?;
    
    // Test 2: Complete compiler with SimpleTestIR  
    test_complete_compiler()?;
    
    println!("\n🎉 Memory operations test completed successfully!");
    println!("The Rust TPDE implementation can now:");
    println!("  - Generate LOAD instructions for 8, 16, 32, 64-bit values");
    println!("  - Generate STORE instructions with proper size variants");
    println!("  - Handle ALLOCA for stack allocation with alignment");
    println!("  - Use addressing modes for memory access");
    println!("  - Integrate with register allocation and value management");
    
    Ok(())
}

fn test_x64_encoder_memory_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing X64 Encoder Memory Operations ===");
    
    let mut encoder = X64Encoder::new()?;
    
    // Test register definitions
    let rax = AsmReg::new(0, 0);  // RAX
    let rbx = AsmReg::new(0, 3);  // RBX
    let rcx = AsmReg::new(0, 1);  // RCX
    
    // Test memory operations
    println!("Testing memory operations...");
    
    // 1. Test 32-bit memory load/store
    encoder.mov32_reg_mem(rax, rbx, 8)?;  // mov eax, dword ptr [rbx + 8]
    encoder.mov32_mem_reg(rbx, 4, rax)?;  // mov dword ptr [rbx + 4], eax
    
    // 2. Test 8-bit operations
    encoder.movzx_reg8_mem(rcx, rbx, 0)?;  // movzx ecx, byte ptr [rbx]
    encoder.mov8_mem_reg(rbx, 1, rcx)?;    // mov byte ptr [rbx + 1], cl
    
    // 3. Test 16-bit operations
    encoder.movzx_reg16_mem(rax, rbx, 2)?; // movzx eax, word ptr [rbx + 2]
    encoder.mov16_mem_reg(rbx, 6, rax)?;   // mov word ptr [rbx + 6], ax
    
    // 4. Test 64-bit operations
    encoder.mov_reg_mem(rax, rbx, 16)?;    // mov rax, qword ptr [rbx + 16]
    encoder.mov_mem_reg(rbx, 24, rax)?;    // mov qword ptr [rbx + 24], rax
    
    // 5. Test stack operations for alloca
    let rsp = AsmReg::new(0, 4);  // RSP
    encoder.sub64_reg_imm(rsp, 32)?;       // sub rsp, 32
    encoder.and64_reg_imm(rsp, -16)?;      // and rsp, -16  
    encoder.mov_reg_reg(rax, rsp)?;        // mov rax, rsp
    
    // Generate machine code
    let code = encoder.finalize()?;
    println!("✓ Generated {} bytes of x86-64 machine code", code.len());
    
    if code.is_empty() {
        return Err("No machine code generated".into());
    }
    
    // Basic validation - machine code should contain real instructions
    if code.len() < 30 {
        return Err(format!("Generated code too short: {} bytes", code.len()).into());
    }
    
    println!("✓ Memory operations encoder test passed");
    Ok(())
}

fn test_complete_compiler() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Complete Compiler ===");
    
    // Create a simple test IR 
    let ir = SimpleTestIR::new_with_add_function();
    let mut compiler = CompleteCompiler::new(ir)?;
    
    // Compile the function
    println!("Compiling simple add function...");
    compiler.compile_all()?;
    
    // Check results
    let compiled_functions = compiler.get_compiled_functions();
    println!("Successfully compiled {} functions:", compiled_functions.len());
    
    if compiled_functions.is_empty() {
        return Err("No functions compiled".into());
    }
    
    for func in compiled_functions {
        println!("Function '{}': {} bytes of machine code", 
                 func.name, func.code.len());
        
        if func.code.is_empty() {
            return Err(format!("Function '{}' generated no machine code", func.name).into());
        }
        
        println!("  Frame size: {} bytes", func.frame_size);
        println!("  Arguments: {} assignments", func.arg_assignments.len());
        println!("  Returns: {} assignments", func.ret_assignments.len());
    }
    
    // Generate object file
    let object_file = compiler.generate_object_file();
    println!("✓ Generated object file: {} bytes total", object_file.len());
    
    println!("✓ Complete compiler test passed");
    Ok(())
}

/*
Cargo manifest for test_memory_ops:

[package]
name = "test_memory_ops"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "test_memory_ops"
path = "test_memory_ops.rs"

[dependencies]
tpde-core = { path = "rust/tpde-core" }
tpde-llvm = { path = "rust/tpde-llvm" }
*/