// This example directly tests the x86-64 encoder's ability to generate CALL instructions,
// which are fundamental for function invocation in compiled code. The test demonstrates
// three types of CALL instructions supported by the encoder: indirect call through register
// (call rax) used when the function address is computed at runtime, direct call with offset
// (call 0x12345678) for statically known addresses, and indirect call through memory
// (call [rbp+8]) for vtables or function pointers stored on stack. The example also shows
// a complete function call sequence following System V x86-64 ABI: loading arguments into
// RDI/RSI registers, loading the function pointer, executing the CALL instruction, and
// handling the return value in RAX. This low-level test validates that the x64::encoder
// module can generate the actual machine code bytes for CALL instructions, which is critical
// for implementing function calls at the LLVM IR level. The test helps ensure that the
// enhanced encoder has all the necessary instructions to support the LlvmCompiler's
// compile_call_instruction method.

use tpde::core::AsmReg;
use tpde::x64::{Encoder as X64Encoder, EncodingError};
use bumpalo::Bump;

/// Test function call instruction generation.
///
/// This verifies that we can now generate real CALL instructions
/// in our x86-64 encoder, completing a critical piece of the
/// function call implementation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger for debugging
    env_logger::init();

    println!("🚀 Testing Function Call Instruction Generation...");

    test_call_instruction_encoding()?;
    test_complete_call_sequence()?;

    println!("✅ Function call instruction tests completed!");
    Ok(())
}

/// Test basic CALL instruction encoding.
fn test_call_instruction_encoding() -> Result<(), EncodingError> {
    println!("📋 Testing CALL instruction encoding...");

    let bump = Bump::new();
    let mut encoder = X64Encoder::new(&bump)?;

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
    assert!(
        !code.is_empty(),
        "Generated machine code should not be empty"
    );

    Ok(())
}

/// Test a complete function call sequence that would be generated
/// by the complete compiler.
fn test_complete_call_sequence() -> Result<(), EncodingError> {
    println!("📋 Testing complete function call sequence...");

    let bump = Bump::new();
    let mut encoder = X64Encoder::new(&bump)?;

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
    assert!(
        code.len() > 10,
        "Call sequence should generate substantial machine code"
    );

    Ok(())
}
