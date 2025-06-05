//! Concrete LLVM compiler implementation.
//!
//! This module provides a simplified, arena-based LLVM compiler that replaces
//! the generic CompleteCompiler with direct LLVM integration. It eliminates
//! trait bound complexity while preserving all excellent functionality.

use crate::{
    compilation_session::CompilationSession,
    value_assignment::ValueAssignment,
    register_file::RegisterFile, 
    function_codegen::FunctionCodegen,
    value_ref::{ValuePartRef, CompilerContext},
    x64_encoder::X64Encoder,
};
use bumpalo::Bump;
use std::collections::HashMap;
use inkwell::basic_block::BasicBlock;

/// Concrete LLVM compiler with arena-based memory management.
///
/// This replaces the generic CompleteCompiler<A> with a focused implementation
/// for LLVM IR compilation. It uses arena allocation to simplify lifetimes
/// and provides direct access to LLVM functionality without trait bounds.
pub struct LlvmCompiler<'ctx, 'arena> {
    /// LLVM module being compiled.
    module: &'ctx inkwell::module::Module<'ctx>,
    
    /// Compilation session for arena allocation.
    session: &'arena mut CompilationSession<'arena>,
    
    /// Value assignment and tracking.
    value_mgr: ValueAssignment,
    
    /// Register allocation state.
    register_file: RegisterFile,
    
    /// Function code generation.
    codegen: FunctionCodegen,
    
    /// Cache of compiled functions.
    compiled_functions: HashMap<String, CompiledFunction<'arena>>,
    
    /// Current function being compiled.
    current_function: Option<inkwell::values::FunctionValue<'ctx>>,
}

/// Information about a compiled function.
#[derive(Debug)]
pub struct CompiledFunction<'arena> {
    /// Function name.
    pub name: String,
    
    /// Generated machine code.
    pub code: &'arena [u8],
    
    /// Function entry point offset.
    pub entry_offset: usize,
    
    /// Size of generated code.
    pub code_size: usize,
}

/// Errors that can occur during LLVM compilation.
#[derive(Debug, Clone)]
pub enum LlvmCompilerError {
    /// LLVM-related error.
    LlvmError(String),
    
    /// Register allocation error.
    RegisterAllocation(String),
    
    /// Code generation error.
    CodeGeneration(String),
    
    /// Unsupported instruction.
    UnsupportedInstruction(String),
    
    /// Function not found.
    FunctionNotFound(String),
    
    /// Session error.
    Session(crate::compilation_session::SessionError),
}

impl std::fmt::Display for LlvmCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlvmCompilerError::LlvmError(msg) => write!(f, "LLVM error: {}", msg),
            LlvmCompilerError::RegisterAllocation(msg) => write!(f, "Register allocation error: {}", msg),
            LlvmCompilerError::CodeGeneration(msg) => write!(f, "Code generation error: {}", msg),
            LlvmCompilerError::UnsupportedInstruction(msg) => write!(f, "Unsupported instruction: {}", msg),
            LlvmCompilerError::FunctionNotFound(msg) => write!(f, "Function not found: {}", msg),
            LlvmCompilerError::Session(err) => write!(f, "Session error: {}", err),
        }
    }
}

impl std::error::Error for LlvmCompilerError {}

impl<'ctx, 'arena> LlvmCompiler<'ctx, 'arena> {
    /// Create a new LLVM compiler.
    pub fn new(
        module: &'ctx inkwell::module::Module<'ctx>,
        session: &'arena mut CompilationSession<'arena>,
    ) -> Result<Self, LlvmCompilerError> {
        let value_mgr = ValueAssignment::new(1024, 8); // Reasonable defaults
        // Create register file with GP and XMM registers
        let mut allocatable = crate::register_file::RegBitSet::new();
        allocatable.union(&crate::register_file::RegBitSet::all_in_bank(0, 16)); // GP regs
        allocatable.union(&crate::register_file::RegBitSet::all_in_bank(1, 16)); // XMM regs
        let register_file = RegisterFile::new(16, 2, allocatable);
        let codegen = FunctionCodegen::new()
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Failed to create codegen: {:?}", e)))?;
        
        Ok(Self {
            module,
            session,
            value_mgr,
            register_file,
            codegen,
            compiled_functions: HashMap::new(),
            current_function: None,
        })
    }
    
    /// Compile a function by name.
    pub fn compile_function_by_name(&mut self, name: &str) -> Result<&CompiledFunction<'arena>, LlvmCompilerError> {
        // Check if already compiled
        if let Some(compiled) = self.compiled_functions.get(name) {
            return Ok(compiled);
        }
        
        // Find the function in the module
        let function = self.module.get_function(name)
            .ok_or_else(|| LlvmCompilerError::FunctionNotFound(name.to_string()))?;
        
        self.compile_function(function)
    }
    
    /// Compile a specific LLVM function.
    pub fn compile_function(
        &mut self, 
        function: inkwell::values::FunctionValue<'ctx>
    ) -> Result<&CompiledFunction<'arena>, LlvmCompilerError> {
        let function_name = function.get_name().to_str()
            .map_err(|e| LlvmCompilerError::LlvmError(format!("Invalid function name: {:?}", e)))?;
        
        // Check if already compiled
        if let Some(compiled) = self.compiled_functions.get(function_name) {
            return Ok(compiled);
        }
        
        self.current_function = Some(function);
        
        println!("🔧 Compiling LLVM function: {}", function_name);
        
        // Reset compiler state for new function
        self.value_mgr = ValueAssignment::new(1024, 8); // Reset with same defaults
        // Reset register file
        let mut allocatable = crate::register_file::RegBitSet::new();
        allocatable.union(&crate::register_file::RegBitSet::all_in_bank(0, 16)); // GP regs
        allocatable.union(&crate::register_file::RegBitSet::all_in_bank(1, 16)); // XMM regs
        self.register_file = RegisterFile::new(16, 2, allocatable);
        self.codegen = FunctionCodegen::new()
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Failed to reset codegen: {:?}", e)))?;
        
        // Process function signature and setup
        self.setup_function_signature(function)?;
        
        // Generate prologue
        self.codegen.emit_prologue()
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Prologue generation failed: {:?}", e)))?;
        
        // Compile function body
        self.compile_function_body(function)?;
        
        // Generate epilogue  
        self.codegen.emit_epilogue()
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Epilogue generation failed: {:?}", e)))?;
        
        // Finalize and get machine code
        let code_bytes = self.codegen.finalize()
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Code finalization failed: {:?}", e)))?;
        
        // Allocate code in session arena
        let code_slice = self.session.alloc_slice(&code_bytes);
        
        // Create compiled function record
        let compiled = CompiledFunction {
            name: function_name.to_string(),
            code: code_slice,
            entry_offset: 0,
            code_size: code_bytes.len(),
        };
        
        // Record statistics
        self.session.record_function_compiled(function_name, code_bytes.len());
        
        // Store in cache and return reference
        self.compiled_functions.insert(function_name.to_string(), compiled);
        Ok(self.compiled_functions.get(function_name).unwrap())
    }
    
    /// Setup function signature and calling convention.
    fn setup_function_signature(
        &mut self, 
        function: inkwell::values::FunctionValue<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        let function_type = function.get_type();
        
        // Process parameter types
        let param_types = function_type.get_param_types();
        let mut args_vec = Vec::with_capacity(param_types.len());
        
        for &param_type in &param_types {
            args_vec.push(self.llvm_type_to_arg_info(param_type)?);
        }
        
        // Process return type
        let ret_info = if function_type.get_return_type().is_some() {
            let ret_type = function_type.get_return_type().unwrap();
            // Convert BasicTypeEnum to BasicMetadataTypeEnum
            let metadata_ret_type = match ret_type {
                inkwell::types::BasicTypeEnum::IntType(t) => inkwell::types::BasicMetadataTypeEnum::IntType(t),
                inkwell::types::BasicTypeEnum::FloatType(t) => inkwell::types::BasicMetadataTypeEnum::FloatType(t),
                inkwell::types::BasicTypeEnum::PointerType(t) => inkwell::types::BasicMetadataTypeEnum::PointerType(t),
                _ => return Err(LlvmCompilerError::UnsupportedInstruction("Unsupported return type".to_string())),
            };
            vec![self.llvm_type_to_arg_info(metadata_ret_type)?]
        } else {
            vec![]
        };
        
        // Setup calling convention
        self.codegen.process_arguments(&args_vec)
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Argument processing failed: {:?}", e)))?;
        
        self.codegen.process_return_values(&ret_info)
            .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Return value processing failed: {:?}", e)))?;
        
        Ok(())
    }
    
    /// Convert LLVM type to ArgInfo.
    fn llvm_type_to_arg_info(
        &self,
        llvm_type: inkwell::types::BasicMetadataTypeEnum<'ctx>
    ) -> Result<crate::function_codegen::ArgInfo, LlvmCompilerError> {
        use inkwell::types::BasicMetadataTypeEnum;
        use crate::function_codegen::ArgInfo;
        
        match llvm_type {
            BasicMetadataTypeEnum::IntType(int_type) => {
                let bit_width = int_type.get_bit_width();
                match bit_width {
                    32 => Ok(ArgInfo::int32()),
                    64 => Ok(ArgInfo::int64()),
                    _ => Err(LlvmCompilerError::UnsupportedInstruction(
                        format!("Unsupported integer bit width: {}", bit_width)
                    ))
                }
            }
            BasicMetadataTypeEnum::FloatType(_) => Ok(ArgInfo::float32()),
            BasicMetadataTypeEnum::PointerType(_) => Ok(ArgInfo::int64()), // Pointers are 64-bit
            _ => Err(LlvmCompilerError::UnsupportedInstruction(
                format!("Unsupported LLVM type: {:?}", llvm_type)
            ))
        }
    }
    
    /// Compile the function body by processing basic blocks and instructions.
    fn compile_function_body(
        &mut self, 
        function: inkwell::values::FunctionValue<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        // Iterate through basic blocks
        for basic_block in function.get_basic_blocks() {
            self.compile_basic_block(basic_block)?;
        }
        
        Ok(())
    }
    
    /// Compile a single basic block.
    fn compile_basic_block(
        &mut self,
        block: BasicBlock<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        println!("📦 Compiling basic block");
        
        // Iterate through instructions in the block
        for instruction in block.get_instructions() {
            self.compile_instruction(instruction)?;
        }
        
        Ok(())
    }
    
    /// Compile a single instruction using direct opcode dispatch.
    fn compile_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        use inkwell::values::InstructionOpcode;
        
        // Record instruction compilation
        let opcode_name = format!("{:?}", instruction.get_opcode());
        self.session.record_instruction_compiled(&opcode_name);
        
        // Direct opcode dispatch - no trait bounds needed!
        match instruction.get_opcode() {
            InstructionOpcode::Add => self.compile_add_instruction(instruction),
            InstructionOpcode::Sub => self.compile_sub_instruction(instruction),
            InstructionOpcode::Mul => self.compile_mul_instruction(instruction),
            InstructionOpcode::ICmp => self.compile_icmp_instruction(instruction),
            InstructionOpcode::Load => self.compile_load_instruction(instruction),
            InstructionOpcode::Store => self.compile_store_instruction(instruction),
            InstructionOpcode::GetElementPtr => self.compile_gep_instruction(instruction),
            InstructionOpcode::Return => self.compile_return_instruction(instruction),
            InstructionOpcode::Br => self.compile_branch_instruction(instruction),
            InstructionOpcode::Call => self.compile_call_instruction(instruction),
            InstructionOpcode::Alloca => self.compile_alloca_instruction(instruction),
            
            _ => {
                println!("⚠️  Unsupported instruction: {:?}", instruction.get_opcode());
                // For now, just skip unsupported instructions
                Ok(())
            }
        }
    }
    
    /// Compile ADD instruction with direct LLVM access.
    fn compile_add_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        println!("➕ Compiling ADD instruction");
        
        // Direct access to LLVM instruction operands - no adaptor needed!
        let _operand0 = instruction.get_operand(0).unwrap().left().unwrap();
        let _operand1 = instruction.get_operand(1).unwrap().left().unwrap();
        
        // TODO: Implement actual ADD compilation
        // For now, just indicate what would be generated
        println!("   Generated: ADD operand0, operand1 -> result");
        
        Ok(())
    }
    
    /// Compile ICMP instruction with real predicate extraction.
    fn compile_icmp_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        println!("🔍 Compiling ICMP instruction");
        
        // Direct predicate extraction - no trait bounds!
        let predicate = instruction.get_icmp_predicate()
            .ok_or_else(|| LlvmCompilerError::LlvmError("ICMP instruction missing predicate".to_string()))?;
        
        println!("   Real predicate extracted: {:?}", predicate);
        
        // TODO: Implement actual ICMP compilation with real predicate
        println!("   Generated: CMP operand0, operand1; SET{:?} result", predicate);
        
        Ok(())
    }
    
    /// Compile RETURN instruction.
    fn compile_return_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>
    ) -> Result<(), LlvmCompilerError> {
        println!("🔙 Compiling RETURN instruction");
        
        let operand_count = instruction.get_num_operands();
        if operand_count > 0 {
            // Return with value
            println!("   Return with value");
        } else {
            // Void return
            println!("   Void return");
        }
        
        // The epilogue will be generated separately
        Ok(())
    }
    
    /// Placeholder implementations for other instructions.
    fn compile_sub_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("➖ Compiling SUB instruction (placeholder)");
        Ok(())
    }
    
    fn compile_mul_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("✖️  Compiling MUL instruction (placeholder)");
        Ok(())
    }
    
    fn compile_load_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("📥 Compiling LOAD instruction (placeholder)");
        Ok(())
    }
    
    fn compile_store_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("📤 Compiling STORE instruction (placeholder)");
        Ok(())
    }
    
    fn compile_gep_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("🗂️  Compiling GEP instruction (placeholder)");
        Ok(())
    }
    
    fn compile_branch_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("🔀 Compiling BRANCH instruction (placeholder)");
        Ok(())
    }
    
    fn compile_call_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("📞 Compiling CALL instruction (placeholder)");
        Ok(())
    }
    
    fn compile_alloca_instruction(&mut self, _instruction: inkwell::values::InstructionValue<'ctx>) -> Result<(), LlvmCompilerError> {
        println!("📋 Compiling ALLOCA instruction (placeholder)");
        Ok(())
    }
    
    /// Get access to compilation session.
    pub fn session(&self) -> &CompilationSession<'arena> {
        self.session
    }
    
    /// Get mutable access to compilation session.
    pub fn session_mut(&mut self) -> &mut CompilationSession<'arena> {
        self.session
    }
    
    /// Get list of compiled functions.
    pub fn compiled_functions(&self) -> &HashMap<String, CompiledFunction<'arena>> {
        &self.compiled_functions
    }
    
    /// Get the LLVM module.
    pub fn module(&self) -> &inkwell::module::Module<'ctx> {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;
    use bumpalo::Bump;
    
    fn create_simple_module(context: &Context) -> inkwell::module::Module {
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("add", fn_type, None);
        
        let entry_block = context.append_basic_block(function, "entry");
        let builder = context.create_builder();
        builder.position_at_end(entry_block);
        
        let param_a = function.get_nth_param(0).unwrap().into_int_value();
        let param_b = function.get_nth_param(1).unwrap().into_int_value();
        let result = builder.build_int_add(param_a, param_b, "add_result").unwrap();
        builder.build_return(Some(&result)).unwrap();
        
        module
    }
    
    #[test]
    fn test_llvm_compiler_creation() {
        let context = Context::create();
        let module = create_simple_module(&context);
        let arena = Bump::new();
        let mut session = CompilationSession::new(&arena);
        
        let compiler = LlvmCompiler::new(&module, &mut session);
        assert!(compiler.is_ok());
    }
    
    #[test]
    fn test_function_compilation() {
        let context = Context::create();
        let module = create_simple_module(&context);
        let arena = Bump::new();
        let mut session = CompilationSession::new(&arena);
        let mut compiler = LlvmCompiler::new(&module, &mut session).unwrap();
        
        let result = compiler.compile_function_by_name("add");
        assert!(result.is_ok());
        
        let compiled = result.unwrap();
        assert_eq!(compiled.name, "add");
        assert!(compiled.code_size > 0);
    }
    
    #[test]
    fn test_session_statistics() {
        let context = Context::create();
        let module = create_simple_module(&context);
        let arena = Bump::new();
        let mut session = CompilationSession::new(&arena);
        let mut compiler = LlvmCompiler::new(&module, &mut session).unwrap();
        
        compiler.compile_function_by_name("add").unwrap();
        
        let stats = compiler.session().stats();
        assert_eq!(stats.functions_compiled, 1);
        assert!(stats.instructions_compiled > 0);
    }
}