use tpde_core::x64_encoder::{X64Encoder, EncodingError};
use tpde_core::register_file::AsmReg;

/// Test function call instruction generation.
///
/// This verifies that we can now generate real CALL instructions
/// in our x86-64 encoder, completing a critical piece of the
/// function call implementation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Function Call Instruction Generation...");
    
    test_call_instruction_encoding()?;
    test_complete_call_sequence()?;
    
    println!("✅ Function call instruction tests completed!");
    Ok(())
}

/// Test basic CALL instruction encoding.
fn test_call_instruction_encoding() -> Result<(), EncodingError> {
    println!("📋 Testing CALL instruction encoding...");
    
    let mut encoder = X64Encoder::new()?;
    
    // Test indirect call through register (call rax)
    let target_reg = AsmReg::new(0, 0); // RAX
    encoder.call_reg(target_reg)?;
    println!("✅ Generated indirect CALL through register RAX");
    
    // Test direct call (call offset)
    encoder.call_direct(0x12345678)?;
    println!("✅ Generated direct CALL with offset");
    
    // Test indirect call through memory (call [rbp+8])
    let base_reg = AsmReg::new(0, 5); // RBP
    encoder.call_mem(base_reg, 8)?;
    println!("✅ Generated indirect CALL through memory [rbp+8]");
    
    // Generate machine code to verify instructions work
    let code = encoder.finalize()?;
    println!("📦 Generated {} bytes of machine code", code.len());
    
    // Verify we have actual machine code (not empty)
    assert!(!code.is_empty(), "Generated machine code should not be empty");
    
    Ok(())
}

/// Test a complete function call sequence that would be generated
/// by the complete compiler.
fn test_complete_call_sequence() -> Result<(), EncodingError> {
    println!("📋 Testing complete function call sequence...");
    
    let mut encoder = X64Encoder::new()?;
    
    // Simulate System V x86-64 calling convention
    let rdi = AsmReg::new(0, 7); // First argument register
    let rsi = AsmReg::new(0, 6); // Second argument register
    let rax = AsmReg::new(0, 0); // Return value register
    let target_reg = AsmReg::new(0, 1); // RCX - function pointer
    
    // Step 1: Move arguments to calling convention registers
    // mov rdi, <arg1>  (would be loaded from value in real scenario)
    encoder.mov_reg_imm(rdi, 42)?;
    
    // mov rsi, <arg2>  (would be loaded from value in real scenario)  
    encoder.mov_reg_imm(rsi, 24)?;
    
    // Step 2: Load function pointer to register
    // mov rcx, <function_ptr>  (would be loaded from IR value)
    encoder.mov_reg_imm(target_reg, 0x12345678)?;
    
    // Step 3: Generate call instruction
    encoder.call_reg(target_reg)?;
    
    // Step 4: Handle return value (already in RAX by ABI)
    // mov <result_location>, rax  (would store to IR result value)
    let result_reg = AsmReg::new(0, 2); // RDX for demonstration
    encoder.mov_reg_reg(result_reg, rax)?;
    
    println!("✅ Generated complete call sequence:");
    println!("   - Argument setup (mov rdi, 42; mov rsi, 24)");
    println!("   - Function pointer load (mov rcx, ptr)");
    println!("   - Function call (call rcx)");
    println!("   - Return value handling (mov rdx, rax)");
    
    // Generate final machine code
    let code = encoder.finalize()?;
    println!("📦 Complete call sequence: {} bytes", code.len());
    
    // This represents what the Rust TPDE can now generate for function calls
    assert!(code.len() > 10, "Call sequence should generate substantial machine code");
    
    Ok(())
}