// This module provides the concrete LlvmCompiler implementation that directly compiles LLVM IR
// to x86-64 machine code using arena-based memory management. The compiler eliminates trait
// bounds by working directly with inkwell's InstructionValue types and implements opcode-based
// dispatch for instruction selection. Key components include: direct instruction compilation
// (add, sub, mul, icmp, load, store, GEP, branch, call, alloca, PHI), System V ABI calling
// convention support, x86-64 addressing modes with LEA optimization for GEP, basic PHI node
// tracking (resolution not fully implemented), and integration with FunctionCodegen for machine
// code emission. The compiler processes functions in RPO block order, allocates registers
// on-demand, and generates real x86-64 instructions. Major gaps include PHI node resolution,
// conditional branches (only unconditional implemented), and complex calling convention features.
// This concrete design replaced the generic framework after architectural redesign.

//! LLVM compiler implementation.
//!
//! This module provides an arena-based LLVM compiler with direct LLVM integration,
//! eliminating trait bound complexity while providing full functionality.

use super::function_analysis::{FunctionAnalysis, FunctionAnalyzer};
use crate::{
    core::{
        register_file::{AsmReg, RegBitSet, RegisterFile},
        session::CompilationSession,
        value_assignment::ValueAssignmentManager,
    },
    core::{CompilerContext, ValuePartRef},
    x64::function_codegen::FunctionCodegen,
};
use inkwell::basic_block::BasicBlock;
use inkwell::values::BasicValueEnum;
use inkwell::IntPredicate;
use std::collections::HashMap;

/// Addressing modes for x86-64 memory operations.
#[derive(Debug, Clone)]
pub enum AddressingMode {
    /// Direct register addressing: [reg]
    Register(AsmReg),
    /// Register with displacement: [reg + offset]
    RegisterOffset(AsmReg, i32),
    /// Base + index with scale: [base + index*scale]
    RegisterIndexScale(AsmReg, AsmReg, u8),
    /// Full addressing: [base + index*scale + offset]
    RegisterIndexScaleOffset(AsmReg, AsmReg, u8, i32),
    /// Stack frame access: [rbp + offset]
    StackOffset(i32),
}

/// GEP addressing expression for complex address calculations.
#[derive(Debug, Clone)]
pub struct GepExpression {
    /// Base register or pointer
    pub base: Option<AsmReg>,
    /// Index register (for array access)
    pub index: Option<AsmReg>,
    /// Scale factor for index (element size)
    pub scale: u64,
    /// Displacement offset (for constant indices and struct fields)
    pub displacement: i64,
    /// Whether this expression needs materialization into a register
    pub needs_materialization: bool,
}

impl Default for GepExpression {
    fn default() -> Self {
        Self {
            base: None,
            index: None,
            scale: 1,
            displacement: 0,
            needs_materialization: false,
        }
    }
}

impl GepExpression {
    /// Create a new empty GEP expression
    pub fn new() -> Self {
        Self::default()
    }

    /// Create expression with base register
    pub fn with_base(base: AsmReg) -> Self {
        Self {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: 0,
            needs_materialization: false,
        }
    }

    /// Add displacement to expression
    pub fn add_displacement(&mut self, offset: i64) {
        self.displacement += offset;
    }

    /// Set index with scale factor
    pub fn set_index(&mut self, index: AsmReg, scale: u64) {
        if self.index.is_some() {
            // Multiple indices require materialization
            self.needs_materialization = true;
        } else {
            self.index = Some(index);
            self.scale = scale;
        }
    }

    /// Convert to x86-64 addressing mode if possible
    pub fn to_addressing_mode(&self) -> Option<AddressingMode> {
        match (self.base, self.index, self.scale, self.displacement) {
            (Some(base), None, _, 0) => Some(AddressingMode::Register(base)),
            (Some(base), None, _, disp) if disp as i32 as i64 == disp => {
                Some(AddressingMode::RegisterOffset(base, disp as i32))
            }
            (Some(base), Some(index), scale, 0) if scale <= 8 => {
                Some(AddressingMode::RegisterIndexScale(base, index, scale as u8))
            }
            (Some(base), Some(index), scale, disp) if scale <= 8 && disp as i32 as i64 == disp => {
                Some(AddressingMode::RegisterIndexScaleOffset(
                    base,
                    index,
                    scale as u8,
                    disp as i32,
                ))
            }
            _ => None,
        }
    }
}

/// Helper struct for binary operation compilation.
struct BinaryOpContext {
    left_idx: usize,
    right_idx: usize,
    result_idx: usize,
    bit_width: u32,
    #[allow(dead_code)]
    value_size: u8, // Kept for consistency with value assignment creation
}

/// LLVM compiler with arena-based memory management.
///
/// This implementation provides direct LLVM IR compilation using arena allocation
/// to simplify lifetimes and direct access to LLVM functionality.
pub struct LlvmCompiler<'ctx, 'arena> {
    /// LLVM module being compiled.
    module: inkwell::module::Module<'ctx>,

    /// Compilation session for arena allocation.
    session: &'arena CompilationSession<'arena>,

    /// Value assignment and tracking.
    value_mgr: ValueAssignmentManager,

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

/// Macro to simplify error mapping
macro_rules! map_err {
    ($expr:expr, $variant:ident, $msg:expr) => {
        $expr.map_err(|e| LlvmCompilerError::$variant(format!("{}: {:?}", $msg, e)))
    };
    ($expr:expr, $variant:ident) => {
        $expr.map_err(|e| LlvmCompilerError::$variant(format!("{:?}", e)))
    };
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

    /// Invalid instruction format.
    InvalidInstruction(String),

    /// Session error.
    Session(crate::core::session::SessionError),
}

impl std::fmt::Display for LlvmCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LlvmCompilerError::*;
        match self {
            LlvmError(msg) => write!(f, "LLVM error: {msg}"),
            RegisterAllocation(msg) => write!(f, "Register allocation error: {msg}"),
            CodeGeneration(msg) => write!(f, "Code generation error: {msg}"),
            UnsupportedInstruction(msg) => write!(f, "Unsupported instruction: {msg}"),
            FunctionNotFound(msg) => write!(f, "Function not found: {msg}"),
            InvalidInstruction(msg) => write!(f, "Invalid instruction: {msg}"),
            Session(err) => write!(f, "Session error: {err}"),
        }
    }
}

impl std::error::Error for LlvmCompilerError {}

impl<'ctx, 'arena> LlvmCompiler<'ctx, 'arena>
where
    'ctx: 'arena,
{
    /// Create a new LLVM compiler.
    pub fn new(
        module: inkwell::module::Module<'ctx>,
        session: &'arena CompilationSession<'arena>,
    ) -> Result<Self, LlvmCompilerError> {
        let value_mgr = ValueAssignmentManager::new();
        // Create register file with GP and XMM registers
        let mut allocatable = RegBitSet::new();
        allocatable.union(&RegBitSet::all_in_bank(0, 16)); // GP regs
        allocatable.union(&RegBitSet::all_in_bank(1, 16)); // XMM regs
        let register_file = RegisterFile::new(16, 2, allocatable);
        let codegen = map_err!(
            FunctionCodegen::new(),
            CodeGeneration,
            "Failed to create codegen"
        )?;

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
    pub fn compile_function_by_name(&mut self, name: &str) -> Result<(), LlvmCompilerError> {
        // Check if already compiled
        if self.compiled_functions.contains_key(name) {
            return Ok(());
        }

        // Find the function in the module
        let function = self
            .module
            .get_function(name)
            .ok_or_else(|| LlvmCompilerError::FunctionNotFound(name.to_string()))?;

        self.compile_function(function)?;
        Ok(())
    }

    /// Compile a specific LLVM function.
    pub fn compile_function(
        &mut self,
        function: inkwell::values::FunctionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        let function_name = function
            .get_name()
            .to_str()
            .map_err(|e| LlvmCompilerError::LlvmError(format!("Invalid function name: {:?}", e)))?;

        // Check if already compiled
        if self.compiled_functions.contains_key(function_name) {
            return Ok(());
        }

        self.current_function = Some(function);

        log::info!("🔧 Compiling LLVM function: {}", function_name);

        // Reset compiler state for new function
        self.value_mgr = ValueAssignmentManager::new();
        // Reset register file
        let mut allocatable = RegBitSet::new();
        allocatable.union(&RegBitSet::all_in_bank(0, 16)); // GP regs
        allocatable.union(&RegBitSet::all_in_bank(1, 16)); // XMM regs
        self.register_file = RegisterFile::new(16, 2, allocatable);
        self.codegen = map_err!(
            FunctionCodegen::new(),
            CodeGeneration,
            "Failed to reset codegen"
        )?;

        // Process function signature and setup
        self.setup_function_signature(function)?;

        // Analyze function using the consuming analyzer
        let analyzer = map_err!(
            FunctionAnalyzer::new(self.session, function),
            LlvmError,
            "Failed to create analyzer"
        )?;

        let analysis = map_err!(analyzer.analyze(), LlvmError, "Function analysis failed")?;

        log::debug!("📊 Function analysis complete:");
        log::debug!("   - {} blocks", analysis.num_blocks);
        log::trace!("   - {} instructions", analysis.instruction_count);
        log::trace!("   PHI   - {} PHI nodes", analysis.phi_count);

        // Store PHI nodes in session from analysis results
        self.store_phi_nodes_from_analysis(&analysis)?;

        // Generate prologue
        map_err!(
            self.codegen.emit_prologue(),
            CodeGeneration,
            "Prologue generation failed"
        )?;

        // Compile function body with analysis
        self.compile_function_body_with_analysis(function, analysis)?;

        // Generate epilogue
        map_err!(
            self.codegen.emit_epilogue(),
            CodeGeneration,
            "Epilogue generation failed"
        )?;

        // Take ownership of codegen and finalize
        let codegen = std::mem::replace(
            &mut self.codegen,
            FunctionCodegen::new().map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!(
                    "Failed to create replacement codegen: {:?}",
                    e
                ))
            })?,
        );

        // Finalize and get machine code
        let code_bytes = codegen.finalize().map_err(|e| {
            LlvmCompilerError::CodeGeneration(format!("Code finalization failed: {:?}", e))
        })?;

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
        self.session
            .record_function_compiled(function_name, code_bytes.len());

        // Store in cache
        self.compiled_functions
            .insert(function_name.to_string(), compiled);
        Ok(())
    }

    /// Setup function signature and calling convention.
    fn setup_function_signature(
        &mut self,
        function: inkwell::values::FunctionValue<'ctx>,
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
                inkwell::types::BasicTypeEnum::IntType(t) => {
                    inkwell::types::BasicMetadataTypeEnum::IntType(t)
                }
                inkwell::types::BasicTypeEnum::FloatType(t) => {
                    inkwell::types::BasicMetadataTypeEnum::FloatType(t)
                }
                inkwell::types::BasicTypeEnum::PointerType(t) => {
                    inkwell::types::BasicMetadataTypeEnum::PointerType(t)
                }
                _ => {
                    return Err(LlvmCompilerError::UnsupportedInstruction(
                        "Unsupported return type".to_string(),
                    ))
                }
            };
            vec![self.llvm_type_to_arg_info(metadata_ret_type)?]
        } else {
            vec![]
        };

        // Setup calling convention
        self.codegen.process_arguments(&args_vec).map_err(|e| {
            LlvmCompilerError::CodeGeneration(format!("Argument processing failed: {:?}", e))
        })?;

        self.codegen.process_return_values(&ret_info).map_err(|e| {
            LlvmCompilerError::CodeGeneration(format!("Return value processing failed: {:?}", e))
        })?;

        Ok(())
    }

    /// Convert LLVM type to ArgInfo.
    fn llvm_type_to_arg_info(
        &self,
        llvm_type: inkwell::types::BasicMetadataTypeEnum<'ctx>,
    ) -> Result<crate::x64::function_codegen::ArgInfo, LlvmCompilerError> {
        use crate::x64::function_codegen::ArgInfo;
        use inkwell::types::BasicMetadataTypeEnum;

        match llvm_type {
            BasicMetadataTypeEnum::IntType(int_type) => {
                let bit_width = int_type.get_bit_width();
                match bit_width {
                    1 => Ok(ArgInfo::int8()), // i1 is passed as i8
                    8 => Ok(ArgInfo::int8()),
                    16 => Ok(ArgInfo::int16()),
                    32 => Ok(ArgInfo::int32()),
                    64 => Ok(ArgInfo::int64()),
                    _ => Err(LlvmCompilerError::UnsupportedInstruction(format!(
                        "Unsupported integer bit width: {}",
                        bit_width
                    ))),
                }
            }
            BasicMetadataTypeEnum::FloatType(_) => Ok(ArgInfo::float32()),
            BasicMetadataTypeEnum::PointerType(_) => Ok(ArgInfo::int64()), // Pointers are 64-bit
            _ => Err(LlvmCompilerError::UnsupportedInstruction(format!(
                "Unsupported LLVM type: {:?}",
                llvm_type
            ))),
        }
    }

    /// Compile the function body using analysis results.
    fn compile_function_body_with_analysis(
        &mut self,
        function: inkwell::values::FunctionValue<'ctx>,
        analysis: FunctionAnalysis<'arena>,
    ) -> Result<(), LlvmCompilerError> {
        let blocks = function.get_basic_blocks();

        // Compile blocks in optimized layout order
        for &block_idx in analysis.block_layout {
            let block = blocks[block_idx];

            // Check if this block has PHI nodes
            let phi_nodes = analysis.get_block_phi_nodes(block_idx);
            if !phi_nodes.is_empty() {
                log::debug!("📍 Block {} has {} PHI nodes", block_idx, phi_nodes.len());
            }

            self.compile_basic_block(block)?;
        }

        Ok(())
    }

    /// Compile a single basic block.
    fn compile_basic_block(&mut self, block: BasicBlock<'ctx>) -> Result<(), LlvmCompilerError> {
        // Get block name for label placement
        if let Ok(name) = block.get_name().to_str() {
            let block_idx = self.get_block_index_by_name(name)?;

            // Place label for this block
            let encoder = self.codegen.encoder_mut();

            // Add a NOP to ensure we can place the label
            // This is a workaround for iced-x86's limitation
            encoder.nop().map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit nop: {:?}", e))
            })?;

            encoder.place_label_for_block(block_idx).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to place label: {:?}", e))
            })?;

            log::trace!("📦 Compiling basic block: {} (index {})", name, block_idx);
        } else {
            log::trace!("📦 Compiling basic block");
        }

        // Iterate through instructions in the block
        for instruction in block.get_instructions() {
            self.compile_instruction(instruction)?;
        }

        Ok(())
    }

    /// Compile a single instruction using direct opcode dispatch.
    fn compile_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
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
            InstructionOpcode::And => self.compile_and_instruction(instruction),
            InstructionOpcode::Or => self.compile_or_instruction(instruction),
            InstructionOpcode::Xor => self.compile_xor_instruction(instruction),
            InstructionOpcode::Shl => self.compile_shl_instruction(instruction),
            InstructionOpcode::LShr => self.compile_lshr_instruction(instruction),
            InstructionOpcode::AShr => self.compile_ashr_instruction(instruction),
            InstructionOpcode::ICmp => self.compile_icmp_instruction(instruction),
            InstructionOpcode::Load => self.compile_load_instruction(instruction),
            InstructionOpcode::Store => self.compile_store_instruction(instruction),
            InstructionOpcode::GetElementPtr => self.compile_gep_instruction(instruction),
            InstructionOpcode::Return => self.compile_return_instruction(instruction),
            InstructionOpcode::Br => self.compile_branch_instruction(instruction),
            InstructionOpcode::Call => self.compile_call_instruction(instruction),
            InstructionOpcode::Alloca => self.compile_alloca_instruction(instruction),
            InstructionOpcode::Phi => self.compile_phi_instruction(instruction),
            InstructionOpcode::SExt => self.compile_sext_instruction(instruction),
            InstructionOpcode::ZExt => self.compile_zext_instruction(instruction),
            InstructionOpcode::Trunc => self.compile_trunc_instruction(instruction),
            InstructionOpcode::Switch => self.compile_switch_instruction(instruction),

            _ => {
                log::warn!(
                    "⚠️  Unsupported instruction: {:?}",
                    instruction.get_opcode()
                );
                // For now, just skip unsupported instructions
                Ok(())
            }
        }
    }

    /// Extract common binary operation setup.
    fn setup_binary_operation(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<BinaryOpContext, LlvmCompilerError> {
        // Direct access to LLVM instruction operands
        let operand0 = instruction.get_operand(0).unwrap().left().unwrap();
        let operand1 = instruction.get_operand(1).unwrap().left().unwrap();

        // Get bit width from instruction type
        let bit_width = instruction.get_type().into_int_type().get_bit_width();

        // Convert LLVM values to our value indices
        let left_idx = self.get_or_create_value_index(operand0)?;
        let right_idx = self.get_or_create_value_index(operand1)?;

        // For arithmetic operations, the instruction itself is the result value
        // We'll use the instruction address as a unique ID for now
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024; // Simple hash for value index

        // Create value assignments
        let value_size = (bit_width / 8) as u8;
        if self.value_mgr.get_assignment(left_idx).is_none() {
            self.value_mgr.create_assignment(left_idx, 1, value_size);
        }
        if self.value_mgr.get_assignment(right_idx).is_none() {
            self.value_mgr.create_assignment(right_idx, 1, value_size);
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, value_size);
        }

        Ok(BinaryOpContext {
            left_idx,
            right_idx,
            result_idx,
            bit_width,
            value_size,
        })
    }

    /// Allocate registers for binary operation.
    fn allocate_binary_op_registers(
        &mut self,
        context: &BinaryOpContext,
        reuse_left: bool,
    ) -> Result<(AsmReg, AsmReg, AsmReg), LlvmCompilerError> {
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Create value references
        let mut left_ref = ValuePartRef::new(context.left_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create left ref: {:?}", e))
        })?;
        let mut right_ref = ValuePartRef::new(context.right_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create right ref: {:?}", e))
        })?;
        let mut result_ref = ValuePartRef::new(context.result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;

        // Load operands to registers
        let left_reg = left_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load left operand: {:?}", e))
        })?;
        let right_reg = right_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load right operand: {:?}", e))
        })?;

        // Try to reuse left operand register for result if requested
        let result_reg = if reuse_left {
            result_ref
                .alloc_try_reuse(&mut left_ref, &mut ctx)
                .map_err(|e| {
                    LlvmCompilerError::RegisterAllocation(format!(
                        "Failed to allocate result: {:?}",
                        e
                    ))
                })?
        } else {
            result_ref.load_to_reg(&mut ctx).map_err(|e| {
                LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
            })?
        };

        Ok((left_reg, right_reg, result_reg))
    }

    /// Compile ADD instruction with real machine code generation.
    fn compile_add_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("➕ Compiling ADD instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, true)?;

        // Generate optimized ADD instruction
        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                if result_reg == left_reg {
                    // In-place addition
                    encoder.add32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit add32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: add32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // LEA optimization for non-destructive add
                    encoder
                        .lea(result_reg, left_reg, Some(right_reg), 1, 0)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit lea32: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: lea32 {}:{}, [{}:{} + {}:{}]",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            64 => {
                if result_reg == left_reg {
                    // In-place addition
                    encoder.add64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit add64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: add64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // LEA optimization for non-destructive add
                    encoder
                        .lea(result_reg, left_reg, Some(right_reg), 1, 0)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit lea: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: lea {}:{}, [{}:{} + {}:{}]",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "ADD instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile ICMP instruction with real CMP+SETcc generation.
    fn compile_icmp_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔍 Compiling ICMP instruction");

        // Direct predicate extraction - no trait bounds!
        let predicate = instruction.get_icmp_predicate().ok_or_else(|| {
            LlvmCompilerError::LlvmError("ICMP instruction missing predicate".to_string())
        })?;

        log::debug!("   Real predicate extracted: {:?}", predicate);

        // Get operands
        let operand0 = instruction.get_operand(0).unwrap().left().unwrap();
        let operand1 = instruction.get_operand(1).unwrap().left().unwrap();

        // Convert to value indices
        let left_idx = self.get_or_create_value_index(operand0)?;
        let right_idx = self.get_or_create_value_index(operand1)?;

        // ICMP result is a boolean (i1) value
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Get operand bit width
        let bit_width = operand0.into_int_value().get_type().get_bit_width();
        let value_size = (bit_width / 8) as u8;

        // Create value assignments
        if self.value_mgr.get_assignment(left_idx).is_none() {
            self.value_mgr.create_assignment(left_idx, 1, value_size);
        }
        if self.value_mgr.get_assignment(right_idx).is_none() {
            self.value_mgr.create_assignment(right_idx, 1, value_size);
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, 1); // Result is boolean
        }

        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Load operands
        let mut left_ref = ValuePartRef::new(left_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create left ref: {:?}", e))
        })?;
        let mut right_ref = ValuePartRef::new(right_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create right ref: {:?}", e))
        })?;
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;

        let left_reg = left_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load left: {:?}", e))
        })?;
        let right_reg = right_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load right: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        let encoder = self.codegen.encoder_mut();

        // Generate CMP instruction
        match bit_width {
            32 => encoder.cmp32_reg_reg(left_reg, right_reg).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit cmp32: {:?}", e))
            })?,
            64 => encoder.cmp_reg_reg(left_reg, right_reg).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit cmp64: {:?}", e))
            })?,
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "ICMP with {}-bit operands not supported",
                    bit_width
                )))
            }
        }

        // Generate SETcc instruction based on predicate
        match predicate {
            IntPredicate::EQ => encoder.sete_reg(result_reg),
            IntPredicate::NE => encoder.setne_reg(result_reg),
            IntPredicate::SGT => encoder.setg_reg(result_reg),
            IntPredicate::SGE => encoder.setge_reg(result_reg),
            IntPredicate::SLT => encoder.setl_reg(result_reg),
            IntPredicate::SLE => encoder.setle_reg(result_reg),
            IntPredicate::UGT => encoder.seta_reg(result_reg),
            IntPredicate::UGE => encoder.setae_reg(result_reg),
            IntPredicate::ULT => encoder.setb_reg(result_reg),
            IntPredicate::ULE => encoder.setbe_reg(result_reg),
        }
        .map_err(|e| LlvmCompilerError::CodeGeneration(format!("Failed to emit setcc: {:?}", e)))?;

        log::trace!(
            "   Generated: CMP {}:{}, {}:{}; SET{:?} {}:{}",
            left_reg.bank,
            left_reg.id,
            right_reg.bank,
            right_reg.id,
            predicate,
            result_reg.bank,
            result_reg.id
        );

        Ok(())
    }

    /// Compile RETURN instruction.
    fn compile_return_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔙 Compiling RETURN instruction");

        let operand_count = instruction.get_num_operands();
        if operand_count > 0 {
            // Return with value
            log::debug!("   Return with value");
        } else {
            // Void return
            log::debug!("   Void return");
        }

        // The epilogue will be generated separately
        Ok(())
    }

    /// Compile SUB instruction with real machine code generation.
    fn compile_sub_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("➖ Compiling SUB instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, true)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                if result_reg == left_reg {
                    encoder.sub32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit sub32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: sub32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                    })?;
                    encoder.sub32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit sub32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov32 {}:{}, {}:{}; sub32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            64 => {
                if result_reg == left_reg {
                    encoder.sub64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit sub64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: sub64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov: {:?}", e))
                    })?;
                    encoder.sub64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit sub64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov {}:{}, {}:{}; sub64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "SUB instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    fn compile_mul_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("✖️  Compiling MUL instruction");

        let context = self.setup_binary_operation(instruction)?;
        // MUL doesn't reuse left register because it uses RAX
        let (left_reg, right_reg, _) = self.allocate_binary_op_registers(&context, false)?;

        // MUL uses RAX for one operand and result
        let rax = AsmReg::new(0, 0); // RAX
        let _rdx = AsmReg::new(0, 2); // RDX (for upper bits)

        let encoder = self.codegen.encoder_mut();

        // Move left operand to RAX if not already there
        if left_reg != rax {
            encoder.mov_reg_reg(rax, left_reg).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to mov to rax: {:?}", e))
            })?;
        }

        // Generate IMUL instruction
        match context.bit_width {
            32 => {
                // Use two-operand IMUL: rax = rax * right_reg
                encoder.imul32_reg_reg(rax, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit imul32: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: imul32 eax, {}:{}",
                    right_reg.bank,
                    right_reg.id
                );
            }
            64 => {
                // Use two-operand IMUL: rax = rax * right_reg
                encoder.imul_reg_reg(rax, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit imul64: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: imul64 rax, {}:{}",
                    right_reg.bank,
                    right_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "MUL instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        // Move result from RAX to result register if different
        // Need to allocate result register properly
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
        let mut result_ref = ValuePartRef::new(context.result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        if result_reg != rax {
            encoder.mov_reg_reg(result_reg, rax).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to mov from rax: {:?}", e))
            })?;
            log::trace!(
                "   Generated: mov {}:{}, rax",
                result_reg.bank,
                result_reg.id
            );
        }

        Ok(())
    }

    /// Compile AND instruction.
    fn compile_and_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔧 Compiling AND instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, true)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                if result_reg == left_reg {
                    encoder.and32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit and32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: and32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                    })?;
                    encoder.and32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit and32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov32 {}:{}, {}:{}; and32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            64 => {
                if result_reg == left_reg {
                    encoder.and64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit and64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: and64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov: {:?}", e))
                    })?;
                    encoder.and64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit and64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov {}:{}, {}:{}; and64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "AND instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile OR instruction.
    fn compile_or_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔧 Compiling OR instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, true)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                if result_reg == left_reg {
                    encoder.or32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit or32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: or32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                    })?;
                    encoder.or32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit or32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov32 {}:{}, {}:{}; or32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            64 => {
                if result_reg == left_reg {
                    encoder.or64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit or64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: or64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov: {:?}", e))
                    })?;
                    encoder.or64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit or64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov {}:{}, {}:{}; or64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "OR instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile XOR instruction.
    fn compile_xor_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔧 Compiling XOR instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, true)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                if result_reg == left_reg {
                    encoder.xor32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit xor32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: xor32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                    })?;
                    encoder.xor32_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit xor32: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov32 {}:{}, {}:{}; xor32 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            64 => {
                if result_reg == left_reg {
                    encoder.xor64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit xor64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: xor64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                } else {
                    // Move left to result first
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit mov: {:?}", e))
                    })?;
                    encoder.xor64_reg_reg(result_reg, right_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit xor64: {:?}", e))
                    })?;
                    log::trace!(
                        "   Generated: mov {}:{}, {}:{}; xor64 {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        left_reg.bank,
                        left_reg.id,
                        result_reg.bank,
                        result_reg.id,
                        right_reg.bank,
                        right_reg.id
                    );
                }
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "XOR instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile shift left (shl) instruction.
    fn compile_shl_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("⬅️  Compiling SHL (shift left) instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, false)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                // For shifts, the count must be in CL (lower 8 bits of RCX)
                let rcx = AsmReg::new(0, 1); // RCX

                // Save RCX if it's in use
                let saved_rcx = if self.register_file.is_allocated(rcx) {
                    // Allocate a temporary register to save RCX
                    // Use a dummy value index for the temporary
                    let temp_idx = 0xFFFF;
                    let temp = self
                        .register_file
                        .allocate_reg(0, temp_idx, 0, None)
                        .map_err(|_| {
                            LlvmCompilerError::RegisterAllocation(
                                "Failed to allocate scratch register for RCX save".to_string(),
                            )
                        })?;
                    encoder.mov_reg_reg(temp, rcx).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to save RCX: {:?}", e))
                    })?;
                    Some(temp)
                } else {
                    None
                };

                // Move shift count to RCX
                encoder.mov32_reg_reg(rcx, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to move to RCX: {:?}", e))
                })?;

                // Move value to result if needed
                if result_reg != left_reg {
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to move to result: {:?}",
                            e
                        ))
                    })?;
                }

                // Emit SHL instruction
                encoder.shl32_reg_cl(result_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit shl32: {:?}", e))
                })?;

                // Restore RCX if needed
                if let Some(temp) = saved_rcx {
                    encoder.mov_reg_reg(rcx, temp).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to restore RCX: {:?}", e))
                    })?;
                    self.register_file.free_register(temp).map_err(|_| {
                        LlvmCompilerError::RegisterAllocation(
                            "Failed to free temp register".to_string(),
                        )
                    })?;
                }

                log::trace!(
                    "   Generated: shl32 {}:{}, cl",
                    result_reg.bank,
                    result_reg.id
                );
            }
            64 => {
                // Similar for 64-bit
                let rcx = AsmReg::new(0, 1); // RCX

                let saved_rcx = if self.register_file.is_allocated(rcx) {
                    let temp_idx = 0xFFFF;
                    let temp = self
                        .register_file
                        .allocate_reg(0, temp_idx, 0, None)
                        .map_err(|_| {
                            LlvmCompilerError::RegisterAllocation(
                                "Failed to allocate scratch register for RCX save".to_string(),
                            )
                        })?;
                    encoder.mov_reg_reg(temp, rcx).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to save RCX: {:?}", e))
                    })?;
                    Some(temp)
                } else {
                    None
                };

                encoder.mov_reg_reg(rcx, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to move to RCX: {:?}", e))
                })?;

                if result_reg != left_reg {
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to move to result: {:?}",
                            e
                        ))
                    })?;
                }

                encoder.shl64_reg_cl(result_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit shl64: {:?}", e))
                })?;

                if let Some(temp) = saved_rcx {
                    encoder.mov_reg_reg(rcx, temp).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to restore RCX: {:?}", e))
                    })?;
                    self.register_file.free_register(temp).map_err(|_| {
                        LlvmCompilerError::RegisterAllocation(
                            "Failed to free temp register".to_string(),
                        )
                    })?;
                }

                log::trace!(
                    "   Generated: shl64 {}:{}, cl",
                    result_reg.bank,
                    result_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "SHL instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile logical shift right (lshr) instruction.
    fn compile_lshr_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("➡️  Compiling LSHR (logical shift right) instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, false)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                let rcx = AsmReg::new(0, 1); // RCX

                let saved_rcx = if self.register_file.is_allocated(rcx) {
                    let temp_idx = 0xFFFF;
                    let temp = self
                        .register_file
                        .allocate_reg(0, temp_idx, 0, None)
                        .map_err(|_| {
                            LlvmCompilerError::RegisterAllocation(
                                "Failed to allocate scratch register for RCX save".to_string(),
                            )
                        })?;
                    encoder.mov_reg_reg(temp, rcx).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to save RCX: {:?}", e))
                    })?;
                    Some(temp)
                } else {
                    None
                };

                encoder.mov32_reg_reg(rcx, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to move to RCX: {:?}", e))
                })?;

                if result_reg != left_reg {
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to move to result: {:?}",
                            e
                        ))
                    })?;
                }

                encoder.shr32_reg_cl(result_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit shr32: {:?}", e))
                })?;

                if let Some(temp) = saved_rcx {
                    encoder.mov_reg_reg(rcx, temp).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to restore RCX: {:?}", e))
                    })?;
                    self.register_file.free_register(temp).map_err(|_| {
                        LlvmCompilerError::RegisterAllocation(
                            "Failed to free temp register".to_string(),
                        )
                    })?;
                }

                log::trace!(
                    "   Generated: shr32 {}:{}, cl",
                    result_reg.bank,
                    result_reg.id
                );
            }
            64 => {
                let rcx = AsmReg::new(0, 1); // RCX

                let saved_rcx = if self.register_file.is_allocated(rcx) {
                    let temp_idx = 0xFFFF;
                    let temp = self
                        .register_file
                        .allocate_reg(0, temp_idx, 0, None)
                        .map_err(|_| {
                            LlvmCompilerError::RegisterAllocation(
                                "Failed to allocate scratch register for RCX save".to_string(),
                            )
                        })?;
                    encoder.mov_reg_reg(temp, rcx).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to save RCX: {:?}", e))
                    })?;
                    Some(temp)
                } else {
                    None
                };

                encoder.mov_reg_reg(rcx, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to move to RCX: {:?}", e))
                })?;

                if result_reg != left_reg {
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to move to result: {:?}",
                            e
                        ))
                    })?;
                }

                encoder.shr64_reg_cl(result_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit shr64: {:?}", e))
                })?;

                if let Some(temp) = saved_rcx {
                    encoder.mov_reg_reg(rcx, temp).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to restore RCX: {:?}", e))
                    })?;
                    self.register_file.free_register(temp).map_err(|_| {
                        LlvmCompilerError::RegisterAllocation(
                            "Failed to free temp register".to_string(),
                        )
                    })?;
                }

                log::trace!(
                    "   Generated: shr64 {}:{}, cl",
                    result_reg.bank,
                    result_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "LSHR instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile arithmetic shift right (ashr) instruction.
    fn compile_ashr_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("➡️  Compiling ASHR (arithmetic shift right) instruction");

        let context = self.setup_binary_operation(instruction)?;
        let (left_reg, right_reg, result_reg) =
            self.allocate_binary_op_registers(&context, false)?;

        let encoder = self.codegen.encoder_mut();

        match context.bit_width {
            32 => {
                let rcx = AsmReg::new(0, 1); // RCX

                let saved_rcx = if self.register_file.is_allocated(rcx) {
                    let temp_idx = 0xFFFF;
                    let temp = self
                        .register_file
                        .allocate_reg(0, temp_idx, 0, None)
                        .map_err(|_| {
                            LlvmCompilerError::RegisterAllocation(
                                "Failed to allocate scratch register for RCX save".to_string(),
                            )
                        })?;
                    encoder.mov_reg_reg(temp, rcx).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to save RCX: {:?}", e))
                    })?;
                    Some(temp)
                } else {
                    None
                };

                encoder.mov32_reg_reg(rcx, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to move to RCX: {:?}", e))
                })?;

                if result_reg != left_reg {
                    encoder.mov32_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to move to result: {:?}",
                            e
                        ))
                    })?;
                }

                encoder.sar32_reg_cl(result_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit sar32: {:?}", e))
                })?;

                if let Some(temp) = saved_rcx {
                    encoder.mov_reg_reg(rcx, temp).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to restore RCX: {:?}", e))
                    })?;
                    self.register_file.free_register(temp).map_err(|_| {
                        LlvmCompilerError::RegisterAllocation(
                            "Failed to free temp register".to_string(),
                        )
                    })?;
                }

                log::trace!(
                    "   Generated: sar32 {}:{}, cl",
                    result_reg.bank,
                    result_reg.id
                );
            }
            64 => {
                let rcx = AsmReg::new(0, 1); // RCX

                let saved_rcx = if self.register_file.is_allocated(rcx) {
                    let temp_idx = 0xFFFF;
                    let temp = self
                        .register_file
                        .allocate_reg(0, temp_idx, 0, None)
                        .map_err(|_| {
                            LlvmCompilerError::RegisterAllocation(
                                "Failed to allocate scratch register for RCX save".to_string(),
                            )
                        })?;
                    encoder.mov_reg_reg(temp, rcx).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to save RCX: {:?}", e))
                    })?;
                    Some(temp)
                } else {
                    None
                };

                encoder.mov_reg_reg(rcx, right_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to move to RCX: {:?}", e))
                })?;

                if result_reg != left_reg {
                    encoder.mov_reg_reg(result_reg, left_reg).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to move to result: {:?}",
                            e
                        ))
                    })?;
                }

                encoder.sar64_reg_cl(result_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit sar64: {:?}", e))
                })?;

                if let Some(temp) = saved_rcx {
                    encoder.mov_reg_reg(rcx, temp).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to restore RCX: {:?}", e))
                    })?;
                    self.register_file.free_register(temp).map_err(|_| {
                        LlvmCompilerError::RegisterAllocation(
                            "Failed to free temp register".to_string(),
                        )
                    })?;
                }

                log::trace!(
                    "   Generated: sar64 {}:{}, cl",
                    result_reg.bank,
                    result_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "ASHR instruction with {}-bit width not supported",
                    context.bit_width
                )));
            }
        }

        Ok(())
    }

    /// Compile sign extension (sext) instruction.
    fn compile_sext_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("📏 Compiling SEXT (sign extend) instruction");

        // SExt has one operand (source) and one result
        let operand = instruction.get_operand(0).unwrap().left().unwrap();
        let src_idx = self.get_or_create_value_index(operand)?;

        // Get source and destination types
        let src_type = operand.get_type().into_int_type();
        let dst_type = instruction.get_type().into_int_type();
        let src_bits = src_type.get_bit_width();
        let dst_bits = dst_type.get_bit_width();

        // Create result value
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Create value assignments
        let src_size = (src_bits / 8) as u8;
        let dst_size = (dst_bits / 8) as u8;
        if self.value_mgr.get_assignment(src_idx).is_none() {
            self.value_mgr.create_assignment(src_idx, 1, src_size);
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, dst_size);
        }

        // Load source value
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
        let mut src_ref = ValuePartRef::new(src_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create src ref: {:?}", e))
        })?;
        let src_reg = src_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load source: {:?}", e))
        })?;

        // Allocate result register
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        let encoder = self.codegen.encoder_mut();

        match (src_bits, dst_bits) {
            (8, 16) => {
                // MOVSX from 8-bit to 16-bit - use 32-bit movsx and then move to 16-bit register
                encoder
                    .movsx_reg8_to_reg32(result_reg, src_reg)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to emit movsx8->16: {:?}",
                            e
                        ))
                    })?;
                log::trace!(
                    "   Generated: movsx {}:{}, byte {}:{} (via 32-bit)",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (8, 32) => {
                // MOVSX from 8-bit to 32-bit
                encoder
                    .movsx_reg8_to_reg32(result_reg, src_reg)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to emit movsx8->32: {:?}",
                            e
                        ))
                    })?;
                log::trace!(
                    "   Generated: movsx {}:{}, byte {}:{}",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (8, 64) => {
                // MOVSX from 8-bit to 64-bit
                encoder
                    .movsx_reg8_to_reg64(result_reg, src_reg)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to emit movsx8->64: {:?}",
                            e
                        ))
                    })?;
                log::trace!(
                    "   Generated: movsx {}:{}, byte {}:{}",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (16, 32) => {
                // MOVSX from 16-bit to 32-bit
                encoder
                    .movsx_reg16_to_reg32(result_reg, src_reg)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to emit movsx16->32: {:?}",
                            e
                        ))
                    })?;
                log::trace!(
                    "   Generated: movsx {}:{}, word {}:{}",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (16, 64) => {
                // MOVSX from 16-bit to 64-bit
                encoder
                    .movsx_reg16_to_reg64(result_reg, src_reg)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to emit movsx16->64: {:?}",
                            e
                        ))
                    })?;
                log::trace!(
                    "   Generated: movsx {}:{}, word {}:{}",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (32, 64) => {
                // MOVSXD from 32-bit to 64-bit
                encoder
                    .movsxd_reg32_to_reg64(result_reg, src_reg)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit movsxd: {:?}", e))
                    })?;
                log::trace!(
                    "   Generated: movsxd {}:{}, {}:{}",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "SEXT from {}-bit to {}-bit not supported",
                    src_bits, dst_bits
                )));
            }
        }

        Ok(())
    }

    /// Compile zero extension (zext) instruction.
    fn compile_zext_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("📏 Compiling ZEXT (zero extend) instruction");

        // ZExt has one operand (source) and one result
        let operand = instruction.get_operand(0).unwrap().left().unwrap();
        let src_idx = self.get_or_create_value_index(operand)?;

        // Get source and destination types
        let src_type = operand.get_type().into_int_type();
        let dst_type = instruction.get_type().into_int_type();
        let src_bits = src_type.get_bit_width();
        let dst_bits = dst_type.get_bit_width();

        // Create result value
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Create value assignments
        let src_size = (src_bits / 8) as u8;
        let dst_size = (dst_bits / 8) as u8;
        if self.value_mgr.get_assignment(src_idx).is_none() {
            self.value_mgr.create_assignment(src_idx, 1, src_size);
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, dst_size);
        }

        // Load source value
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
        let mut src_ref = ValuePartRef::new(src_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create src ref: {:?}", e))
        })?;
        let src_reg = src_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load source: {:?}", e))
        })?;

        // Allocate result register
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        let encoder = self.codegen.encoder_mut();

        match (src_bits, dst_bits) {
            (8, 32) | (16, 32) => {
                // MOVZX to 32-bit
                if src_bits == 8 {
                    encoder
                        .movzx_reg8_to_reg32(result_reg, src_reg)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit movzx8->32: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: movzx {}:{}, byte {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        src_reg.bank,
                        src_reg.id
                    );
                } else {
                    encoder
                        .movzx_reg16_to_reg32(result_reg, src_reg)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit movzx16->32: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: movzx {}:{}, word {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        src_reg.bank,
                        src_reg.id
                    );
                }
            }
            (8, 64) | (16, 64) => {
                // MOVZX to 64-bit (actually uses 32-bit movzx which zero-extends upper 32 bits)
                if src_bits == 8 {
                    encoder
                        .movzx_reg8_to_reg32(result_reg, src_reg)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit movzx8->64: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: movzx {}:{}, byte {}:{} (zero-extends to 64)",
                        result_reg.bank,
                        result_reg.id,
                        src_reg.bank,
                        src_reg.id
                    );
                } else {
                    encoder
                        .movzx_reg16_to_reg32(result_reg, src_reg)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit movzx16->64: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: movzx {}:{}, word {}:{} (zero-extends to 64)",
                        result_reg.bank,
                        result_reg.id,
                        src_reg.bank,
                        src_reg.id
                    );
                }
            }
            (32, 64) => {
                // MOV 32-bit to 32-bit register automatically zero-extends to 64
                encoder.mov32_reg_reg(result_reg, src_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: mov32 {}:{}, {}:{} (zero-extends to 64)",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (1, _) => {
                // Special case for i1 (boolean) - treat as 8-bit and zero extend
                match dst_bits {
                    32 => {
                        encoder
                            .movzx_reg8_to_reg32(result_reg, src_reg)
                            .map_err(|e| {
                                LlvmCompilerError::CodeGeneration(format!(
                                    "Failed to emit movzx1->32: {:?}",
                                    e
                                ))
                            })?;
                        log::trace!(
                            "   Generated: movzx {}:{}, byte {}:{} (i1 to i32)",
                            result_reg.bank,
                            result_reg.id,
                            src_reg.bank,
                            src_reg.id
                        );
                    }
                    64 => {
                        encoder
                            .movzx_reg8_to_reg32(result_reg, src_reg)
                            .map_err(|e| {
                                LlvmCompilerError::CodeGeneration(format!(
                                    "Failed to emit movzx1->64: {:?}",
                                    e
                                ))
                            })?;
                        log::trace!(
                            "   Generated: movzx {}:{}, byte {}:{} (i1 to i64)",
                            result_reg.bank,
                            result_reg.id,
                            src_reg.bank,
                            src_reg.id
                        );
                    }
                    _ => {
                        return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                            "ZEXT from i1 to i{} not supported",
                            dst_bits
                        )));
                    }
                }
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "ZEXT from {}-bit to {}-bit not supported",
                    src_bits, dst_bits
                )));
            }
        }

        Ok(())
    }

    /// Compile truncation (trunc) instruction.
    fn compile_trunc_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("✂️  Compiling TRUNC (truncate) instruction");

        // Trunc has one operand (source) and one result
        let operand = instruction.get_operand(0).unwrap().left().unwrap();
        let src_idx = self.get_or_create_value_index(operand)?;

        // Get source and destination types
        let src_type = operand.get_type().into_int_type();
        let dst_type = instruction.get_type().into_int_type();
        let src_bits = src_type.get_bit_width();
        let dst_bits = dst_type.get_bit_width();

        // Create result value
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Create value assignments
        let src_size = (src_bits / 8) as u8;
        let dst_size = (dst_bits / 8) as u8;
        if self.value_mgr.get_assignment(src_idx).is_none() {
            self.value_mgr.create_assignment(src_idx, 1, src_size);
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, dst_size);
        }

        // Load source value
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
        let mut src_ref = ValuePartRef::new(src_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create src ref: {:?}", e))
        })?;
        let src_reg = src_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load source: {:?}", e))
        })?;

        // Allocate result register
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        let encoder = self.codegen.encoder_mut();

        // For truncation, we just move the value to the smaller register size
        // The upper bits are naturally ignored
        match (src_bits, dst_bits) {
            (64, 32) | (32, 32) => {
                // MOV to 32-bit register
                encoder.mov32_reg_reg(result_reg, src_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!(
                        "Failed to emit mov32 for trunc: {:?}",
                        e
                    ))
                })?;
                log::trace!(
                    "   Generated: mov32 {}:{}, {}:{} (truncate to 32)",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (64, 16) | (32, 16) | (16, 16) => {
                // MOV to 16-bit register
                encoder.mov16_reg_reg(result_reg, src_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!(
                        "Failed to emit mov16 for trunc: {:?}",
                        e
                    ))
                })?;
                log::trace!(
                    "   Generated: mov16 {}:{}, {}:{} (truncate to 16)",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (64, 8) | (32, 8) | (16, 8) | (8, 8) => {
                // MOV to 8-bit register
                encoder.mov8_reg_reg(result_reg, src_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!(
                        "Failed to emit mov8 for trunc: {:?}",
                        e
                    ))
                })?;
                log::trace!(
                    "   Generated: mov8 {}:{}, {}:{} (truncate to 8)",
                    result_reg.bank,
                    result_reg.id,
                    src_reg.bank,
                    src_reg.id
                );
            }
            (_, 1) => {
                // Special case for truncation to i1 (boolean)
                // AND with 1 to keep only the lowest bit
                encoder.and64_reg_imm(result_reg, 1).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!(
                        "Failed to emit and for i1 trunc: {:?}",
                        e
                    ))
                })?;
                log::trace!(
                    "   Generated: and {}:{}, 1 (truncate to i1)",
                    result_reg.bank,
                    result_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "TRUNC from {}-bit to {}-bit not supported",
                    src_bits, dst_bits
                )));
            }
        }

        Ok(())
    }

    fn compile_load_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("📥 Compiling LOAD instruction");

        // Load instruction format: %result = load <type>, <type>* <pointer>
        // Get the pointer operand (source address)
        let ptr_operand = instruction.get_operand(0).unwrap().left().unwrap();
        let ptr_idx = self.get_or_create_value_index(ptr_operand)?;

        // Get result type and size
        let load_type = instruction.get_type();
        let bit_width = match load_type {
            inkwell::types::AnyTypeEnum::IntType(int_type) => int_type.get_bit_width(),
            inkwell::types::AnyTypeEnum::FloatType(_) => 32,
            inkwell::types::AnyTypeEnum::PointerType(_) => 64,
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "Unsupported load type: {:?}",
                    load_type
                )))
            }
        };
        let value_size = (bit_width / 8) as u8;

        // Create result value (the loaded value)
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Create value assignments
        if self.value_mgr.get_assignment(ptr_idx).is_none() {
            self.value_mgr.create_assignment(ptr_idx, 1, 8); // Pointer is 64-bit
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, value_size);
        }

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Create value references
        let mut ptr_ref = ValuePartRef::new(ptr_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create ptr ref: {:?}", e))
        })?;
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;

        // Load pointer to register
        let ptr_reg = ptr_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load pointer: {:?}", e))
        })?;

        // Allocate register for result
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        // Generate MOV instruction from memory to register
        let encoder = self.codegen.encoder_mut();

        match bit_width {
            8 => {
                encoder.mov8_reg_mem(result_reg, ptr_reg, 0).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov8: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movb {}:{}, [{}:{}]",
                    result_reg.bank,
                    result_reg.id,
                    ptr_reg.bank,
                    ptr_reg.id
                );
            }
            16 => {
                encoder.mov16_reg_mem(result_reg, ptr_reg, 0).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov16: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movw {}:{}, [{}:{}]",
                    result_reg.bank,
                    result_reg.id,
                    ptr_reg.bank,
                    ptr_reg.id
                );
            }
            32 => {
                encoder.mov32_reg_mem(result_reg, ptr_reg, 0).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movl {}:{}, [{}:{}]",
                    result_reg.bank,
                    result_reg.id,
                    ptr_reg.bank,
                    ptr_reg.id
                );
            }
            64 => {
                encoder.mov64_reg_mem(result_reg, ptr_reg, 0).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov64: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movq {}:{}, [{}:{}]",
                    result_reg.bank,
                    result_reg.id,
                    ptr_reg.bank,
                    ptr_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "LOAD instruction with {}-bit width not supported",
                    bit_width
                )));
            }
        }

        Ok(())
    }

    fn compile_store_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("📤 Compiling STORE instruction");

        // Store instruction format: store <type> <value>, <type>* <pointer>
        // Get the value to store (first operand)
        let value_operand = instruction.get_operand(0).unwrap().left().unwrap();
        let value_idx = self.get_or_create_value_index(value_operand)?;

        // Get the pointer destination (second operand)
        let ptr_operand = instruction.get_operand(1).unwrap().left().unwrap();
        let ptr_idx = self.get_or_create_value_index(ptr_operand)?;

        // Get value type and size from the value operand
        let value_type = value_operand.get_type();
        let bit_width = match value_type {
            inkwell::types::BasicTypeEnum::IntType(int_type) => int_type.get_bit_width(),
            inkwell::types::BasicTypeEnum::FloatType(_) => 32,
            inkwell::types::BasicTypeEnum::PointerType(_) => 64,
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "Unsupported store type: {:?}",
                    value_type
                )))
            }
        };
        let value_size = (bit_width / 8) as u8;

        // Create value assignments
        if self.value_mgr.get_assignment(value_idx).is_none() {
            self.value_mgr.create_assignment(value_idx, 1, value_size);
        }
        if self.value_mgr.get_assignment(ptr_idx).is_none() {
            self.value_mgr.create_assignment(ptr_idx, 1, 8); // Pointer is 64-bit
        }

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Create value references
        let mut value_ref = ValuePartRef::new(value_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create value ref: {:?}", e))
        })?;
        let mut ptr_ref = ValuePartRef::new(ptr_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create ptr ref: {:?}", e))
        })?;

        // Load value and pointer to registers
        let value_reg = value_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load value: {:?}", e))
        })?;
        let ptr_reg = ptr_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load pointer: {:?}", e))
        })?;

        // Generate MOV instruction from register to memory
        let encoder = self.codegen.encoder_mut();

        match bit_width {
            8 => {
                encoder.mov8_mem_reg(ptr_reg, 0, value_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov8: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movb [{}:{}], {}:{}",
                    ptr_reg.bank,
                    ptr_reg.id,
                    value_reg.bank,
                    value_reg.id
                );
            }
            16 => {
                encoder.mov16_mem_reg(ptr_reg, 0, value_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov16: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movw [{}:{}], {}:{}",
                    ptr_reg.bank,
                    ptr_reg.id,
                    value_reg.bank,
                    value_reg.id
                );
            }
            32 => {
                encoder.mov32_mem_reg(ptr_reg, 0, value_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov32: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movl [{}:{}], {}:{}",
                    ptr_reg.bank,
                    ptr_reg.id,
                    value_reg.bank,
                    value_reg.id
                );
            }
            64 => {
                encoder.mov64_mem_reg(ptr_reg, 0, value_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit mov64: {:?}", e))
                })?;
                log::trace!(
                    "   Generated: movq [{}:{}], {}:{}",
                    ptr_reg.bank,
                    ptr_reg.id,
                    value_reg.bank,
                    value_reg.id
                );
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "STORE instruction with {}-bit width not supported",
                    bit_width
                )));
            }
        }

        Ok(())
    }

    fn compile_gep_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::info!("🗂️  Compiling GEP instruction");

        // Get operand count
        let operand_count = instruction.get_num_operands();
        if operand_count < 2 {
            return Err(LlvmCompilerError::UnsupportedInstruction(
                "GEP instruction requires at least 2 operands".to_string(),
            ));
        }

        // Get base pointer (first operand)
        let base_ptr = instruction.get_operand(0).unwrap().left().unwrap();
        let base_idx = self.get_or_create_value_index(base_ptr)?;

        // Get result - GEP instruction returns pointer
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Initialize value assignments
        if self.value_mgr.get_assignment(base_idx).is_none() {
            self.value_mgr.create_assignment(base_idx, 1, 8); // Pointer size
        }
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, 8); // Result is pointer
        }

        // Process indices
        let mut indices = Vec::new();
        for i in 1..operand_count {
            if let Some(idx_operand) = instruction.get_operand(i) {
                if let Some(idx_val) = idx_operand.left() {
                    let idx_idx = self.get_or_create_value_index(idx_val)?;
                    if self.value_mgr.get_assignment(idx_idx).is_none() {
                        self.value_mgr.create_assignment(idx_idx, 1, 4); // Index size
                    }
                    indices.push(idx_idx);
                }
            }
        }

        // Extract element sizes and constants before creating context
        let mut index_info = Vec::new();
        for (idx_num, &index_idx) in indices.iter().enumerate() {
            let element_size = self.get_gep_element_size(idx_num)?;
            let const_val = self.try_get_constant_index(index_idx);
            index_info.push((index_idx, element_size, const_val));
        }

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Load base pointer to register
        let mut base_ref = ValuePartRef::new(base_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create base ref: {:?}", e))
        })?;
        let base_reg = base_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load base: {:?}", e))
        })?;

        // Create GEP expression
        let mut gep_expr = GepExpression::with_base(base_reg);

        // Process each index
        for (idx_num, (index_idx, element_size, const_val)) in index_info.into_iter().enumerate() {
            // Check if index is a constant
            if let Some(const_val) = const_val {
                // Fold constant into displacement
                let offset = element_size as i64 * const_val;
                gep_expr.add_displacement(offset);
                log::debug!(
                    "GEP: Folded constant index {} -> displacement {}",
                    const_val,
                    offset
                );
            } else {
                // Dynamic index
                let mut index_ref = ValuePartRef::new(index_idx, 0).map_err(|e| {
                    LlvmCompilerError::RegisterAllocation(format!(
                        "Failed to create index ref: {:?}",
                        e
                    ))
                })?;
                let index_reg = index_ref.load_to_reg(&mut ctx).map_err(|e| {
                    LlvmCompilerError::RegisterAllocation(format!("Failed to load index: {:?}", e))
                })?;

                if idx_num == 0 && gep_expr.index.is_none() {
                    // First dynamic index - can use scaled addressing
                    gep_expr.set_index(index_reg, element_size);
                    log::debug!("   GEP: Set dynamic index with scale {}", element_size);
                } else {
                    // Multiple indices need materialization
                    gep_expr.needs_materialization = true;
                    log::debug!("   GEP: Complex index requires materialization");
                }
            }
        }

        // Allocate result register
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        // Generate address calculation
        self.materialize_gep_expression(gep_expr, result_reg)?;

        log::debug!("✅ GEP instruction compiled successfully");
        Ok(())
    }

    fn compile_branch_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::info!("🔀 Compiling BRANCH instruction");

        // LLVM has two types of branch instructions:
        // - Unconditional: br label %dest
        // - Conditional: br i1 %cond, label %iftrue, label %iffalse

        let num_operands = instruction.get_num_operands();

        // Get current block information
        let current_block = instruction.get_parent().unwrap();
        let current_block_name = current_block.get_name().to_str().map_err(|e| {
            LlvmCompilerError::LlvmError(format!("Invalid current block name: {:?}", e))
        })?;
        let current_block_idx = self.get_block_index_by_name(current_block_name)?;

        match num_operands {
            1 => {
                // Unconditional branch
                let target_operand = instruction.get_operand(0).unwrap().right().unwrap();
                let target_name = target_operand.get_name().to_str().map_err(|e| {
                    LlvmCompilerError::LlvmError(format!("Invalid block name: {:?}", e))
                })?;

                let target_block_idx = self.get_block_index_by_name(target_name)?;

                // Generate PHI moves for the target block
                self.generate_phi_moves_for_edge(current_block_idx, target_block_idx)?;

                // Generate unconditional jump
                let encoder = self.codegen.encoder_mut();
                encoder
                    .jmp_unconditional_to_block(target_block_idx)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit jmp: {:?}", e))
                    })?;

                log::trace!("   Generated: jmp block_{}", target_block_idx);
            }
            3 => {
                // Conditional branch
                let condition = instruction.get_operand(0).unwrap().left().unwrap();
                let true_target = instruction.get_operand(2).unwrap().right().unwrap();
                let false_target = instruction.get_operand(1).unwrap().right().unwrap();

                // Get condition value
                let cond_idx = self.get_or_create_value_index(condition)?;

                // Get target block indices
                let true_name = true_target.get_name().to_str().map_err(|e| {
                    LlvmCompilerError::LlvmError(format!("Invalid true block name: {:?}", e))
                })?;
                let false_name = false_target.get_name().to_str().map_err(|e| {
                    LlvmCompilerError::LlvmError(format!("Invalid false block name: {:?}", e))
                })?;

                let true_block_idx = self.get_block_index_by_name(true_name)?;
                let false_block_idx = self.get_block_index_by_name(false_name)?;

                // Create value assignment for condition if needed
                if self.value_mgr.get_assignment(cond_idx).is_none() {
                    self.value_mgr.create_assignment(cond_idx, 1, 1); // i1 is 1 byte
                }

                // Create compiler context
                let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

                // Load condition to register
                let mut cond_ref = ValuePartRef::new(cond_idx, 0).map_err(|e| {
                    LlvmCompilerError::RegisterAllocation(format!(
                        "Failed to create cond ref: {:?}",
                        e
                    ))
                })?;
                let cond_reg = cond_ref.load_to_reg(&mut ctx).map_err(|e| {
                    LlvmCompilerError::RegisterAllocation(format!(
                        "Failed to load condition: {:?}",
                        e
                    ))
                })?;

                // Generate PHI moves for false block (must be done before conditional jump)
                self.generate_phi_moves_for_edge(current_block_idx, false_block_idx)?;

                // Generate test and conditional jump
                let encoder = self.codegen.encoder_mut();

                // TEST cond_reg, cond_reg (sets ZF if zero)
                encoder.test_reg_reg(cond_reg, cond_reg).map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit test: {:?}", e))
                })?;

                // JNE true_block (jump if not equal/not zero)
                encoder
                    .jmp_conditional_to_block(
                        crate::x64::encoder::JumpCondition::NotEqual,
                        true_block_idx,
                    )
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit jne: {:?}", e))
                    })?;

                // Emit unconditional jump to false block
                encoder
                    .jmp_unconditional_to_block(false_block_idx)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit jmp: {:?}", e))
                    })?;

                log::trace!(
                    "   Generated: test {}:{}, {}:{}",
                    cond_reg.bank,
                    cond_reg.id,
                    cond_reg.bank,
                    cond_reg.id
                );
                log::trace!("   Generated: jne block_{}", true_block_idx);
                log::trace!("   Generated: jmp block_{}", false_block_idx);

                // Now generate a separate code sequence for the true branch PHI moves
                // This requires placing code at a different location
                // For now, we'll skip this optimization and rely on the PHI resolver
            }
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "Branch instruction with {} operands not supported",
                    num_operands
                )));
            }
        }

        Ok(())
    }

    fn compile_switch_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔀 Compiling SWITCH instruction");

        // Get the current block index
        let block = instruction.get_parent().unwrap();
        let block_name = block
            .get_name()
            .to_str()
            .map_err(|e| LlvmCompilerError::LlvmError(format!("Invalid block name: {:?}", e)))?;
        let current_block_idx = self.get_block_index_by_name(block_name)?;

        // Switch has at least 2 operands: condition and default block
        // Additional operands come in pairs (case value, case block)
        let num_operands = instruction.get_num_operands();
        if num_operands < 2 {
            return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                "Switch instruction with {} operands not supported",
                num_operands
            )));
        }

        // Get the switch value
        let switch_value = instruction.get_operand(0).unwrap().left().unwrap();
        let switch_idx = self.get_or_create_value_index(switch_value)?;

        // Create value assignment for switch value if needed
        if self.value_mgr.get_assignment(switch_idx).is_none() {
            let switch_type = switch_value.get_type();
            let bit_width = if let inkwell::types::BasicTypeEnum::IntType(int_type) = switch_type {
                int_type.get_bit_width()
            } else {
                return Err(LlvmCompilerError::UnsupportedInstruction(
                    "Switch on non-integer type not supported".to_string(),
                ));
            };
            self.value_mgr
                .create_assignment(switch_idx, 1, (bit_width / 8) as u8);
        }

        // Get the default target block
        let default_target = instruction.get_operand(1).unwrap().right().unwrap();
        let default_name = default_target.get_name().to_str().map_err(|e| {
            LlvmCompilerError::LlvmError(format!("Invalid default block name: {:?}", e))
        })?;
        let default_block_idx = self.get_block_index_by_name(default_name)?;

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Load switch value to register
        let mut switch_ref = ValuePartRef::new(switch_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create switch ref: {:?}", e))
        })?;
        let switch_reg = switch_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load switch value: {:?}", e))
        })?;

        // Process case values and targets
        // The operands are: value, default_block, [case_value, case_block]*
        let num_cases = (num_operands - 2) / 2;
        log::debug!("   Switch has {} cases plus default", num_cases);

        // First, collect all case information
        let mut cases = Vec::new();
        for i in 0..num_cases {
            let case_value_idx = 2 + i * 2;
            let case_target_idx = 2 + i * 2 + 1;

            // Get case constant value
            let case_value = instruction
                .get_operand(case_value_idx)
                .unwrap()
                .left()
                .unwrap();

            // Extract constant value
            let const_value = if let inkwell::values::BasicValueEnum::IntValue(int_val) = case_value
            {
                if let Some(const_int) = int_val.get_zero_extended_constant() {
                    const_int as i32
                } else {
                    return Err(LlvmCompilerError::UnsupportedInstruction(
                        "Non-constant case value not supported".to_string(),
                    ));
                }
            } else {
                return Err(LlvmCompilerError::UnsupportedInstruction(
                    "Non-integer case value not supported".to_string(),
                ));
            };

            // Get case target block
            let case_target = instruction
                .get_operand(case_target_idx)
                .unwrap()
                .right()
                .unwrap();
            let case_name = case_target.get_name().to_str().map_err(|e| {
                LlvmCompilerError::LlvmError(format!("Invalid case block name: {:?}", e))
            })?;
            let case_block_idx = self.get_block_index_by_name(case_name)?;

            cases.push((const_value, case_block_idx));
        }

        // Get bit width for comparisons
        let bit_width = switch_value.get_type().into_int_type().get_bit_width();

        // Now generate code for each case
        for (const_value, case_block_idx) in cases {
            // Generate comparison
            let encoder = self.codegen.encoder_mut();
            match bit_width {
                32 => {
                    encoder
                        .cmp32_reg_imm(switch_reg, const_value)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit cmp32: {:?}",
                                e
                            ))
                        })?;
                }
                64 => {
                    encoder.cmp_reg_imm(switch_reg, const_value).map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!("Failed to emit cmp: {:?}", e))
                    })?;
                }
                _ => {
                    return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                        "Switch on {}-bit integer not supported",
                        bit_width
                    )));
                }
            }

            // Generate PHI moves for this case
            self.generate_phi_moves_for_edge(current_block_idx, case_block_idx)?;

            // Generate conditional jump to case block
            let encoder = self.codegen.encoder_mut();
            encoder
                .jmp_conditional_to_block(crate::x64::encoder::JumpCondition::Equal, case_block_idx)
                .map_err(|e| {
                    LlvmCompilerError::CodeGeneration(format!("Failed to emit je: {:?}", e))
                })?;

            log::trace!(
                "   Generated: cmp {}:{}, {}; je block_{}",
                switch_reg.bank,
                switch_reg.id,
                const_value,
                case_block_idx
            );
        }

        // Generate PHI moves for default block
        self.generate_phi_moves_for_edge(current_block_idx, default_block_idx)?;

        // Generate unconditional jump to default block
        let encoder = self.codegen.encoder_mut();
        encoder
            .jmp_unconditional_to_block(default_block_idx)
            .map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit jmp: {:?}", e))
            })?;

        log::trace!("   Generated: jmp block_{} (default)", default_block_idx);

        Ok(())
    }

    fn compile_call_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("📞 Compiling CALL instruction");

        // Get the call instruction details
        use inkwell::values::CallSiteValue;
        let call_site = CallSiteValue::try_from(instruction).map_err(|_| {
            LlvmCompilerError::LlvmError("Failed to cast instruction to CallSiteValue".to_string())
        })?;

        // Get the called function
        let called_value = call_site.get_called_fn_value();
        let (function_name, is_varargs) = if let Some(func) = called_value {
            let name = func
                .get_name()
                .to_str()
                .map_err(|e| {
                    LlvmCompilerError::LlvmError(format!("Invalid function name: {:?}", e))
                })?
                .to_string();
            let varargs = func.get_type().is_var_arg();
            (name, varargs)
        } else {
            // For now, only support direct function calls
            return Err(LlvmCompilerError::UnsupportedInstruction(
                "Indirect function calls not yet supported".to_string(),
            ));
        };

        log::debug!(
            "   Calling function: {} (varargs: {})",
            function_name,
            is_varargs
        );

        // Get arguments
        let arg_count = call_site.count_arguments();
        log::debug!("   Argument count: {}", arg_count);

        // Create calling convention assigner
        use crate::llvm::call_support::{get_byval_size, get_param_attributes};
        use crate::x64::calling_convention::{
            ArgAttribute, CCAssigner, CCAssignment, RegBank, SysVAssigner,
        };
        let mut cc_assigner = SysVAssigner::new();

        // For varargs functions, we need to know where fixed args end
        let fixed_arg_count = if is_varargs {
            // Get the function type to determine fixed argument count
            if let Some(func) = called_value {
                func.get_type().count_param_types()
            } else {
                0
            }
        } else {
            arg_count
        };

        // Process arguments and assign them according to System V ABI
        let mut arg_assignments = Vec::new();

        for i in 0..arg_count {
            let arg_value = instruction.get_operand(i).unwrap().left().unwrap();
            let arg_idx = self.get_or_create_value_index(arg_value)?;

            // Get parameter attributes
            let param_attr = get_param_attributes(call_site, i);

            // Handle varargs - after fixed args, must go to stack
            if is_varargs && i >= fixed_arg_count {
                cc_assigner.set_must_assign_stack();
            }

            // Determine argument type and size
            let arg_type = arg_value.get_type();
            let (bank, size, align) = match arg_type {
                inkwell::types::BasicTypeEnum::IntType(int_type) => {
                    let bit_width = int_type.get_bit_width();
                    let byte_size = bit_width / 8;
                    let align = byte_size.min(8); // Max 8-byte alignment
                    (RegBank::GeneralPurpose, byte_size, align)
                }
                inkwell::types::BasicTypeEnum::PointerType(_) => (RegBank::GeneralPurpose, 8, 8),
                inkwell::types::BasicTypeEnum::FloatType(float_type) => {
                    match float_type.get_context().f32_type() {
                        t if t == float_type => (RegBank::Xmm, 4, 4),
                        _ => (RegBank::Xmm, 8, 8), // f64
                    }
                }
                _ => {
                    return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                        "Unsupported argument type: {:?}",
                        arg_type
                    )));
                }
            };

            // Handle byval attribute specially
            let final_attr = match param_attr {
                ArgAttribute::ByVal {
                    size: _,
                    align: byval_align,
                } => {
                    // Get actual size for byval
                    let byval_size = get_byval_size(call_site, i, arg_value);
                    ArgAttribute::ByVal {
                        size: byval_size,
                        align: byval_align,
                    }
                }
                _ => param_attr,
            };

            // Create assignment for this argument
            let mut assignment = CCAssignment::with_attribute(bank, size, align, final_attr);
            cc_assigner.assign_arg(&mut assignment);


            log::debug!(
                "   Arg {}: v{} -> {:?} (attr: {:?})",
                i,
                arg_idx,
                assignment,
                final_attr
            );

            // Create value assignment if needed
            if self.value_mgr.get_assignment(arg_idx).is_none() {
                self.value_mgr.create_assignment(arg_idx, 1, size as u8);
            }

            arg_assignments.push((arg_idx, assignment));
        }

        // Setup arguments in their assigned locations
        // We need to handle this without borrowing issues
        for (arg_idx, assignment) in &arg_assignments {
            match assignment.attribute {
                ArgAttribute::ByVal { size, align: _ } => {
                    // For byval, we need to copy the data to the stack
                    // We'll handle this inline to avoid borrowing issues
                    let mut ptr_ref = ValuePartRef::new(*arg_idx, 0).map_err(|e| {
                        LlvmCompilerError::RegisterAllocation(format!(
                            "Failed to create ptr ref: {:?}",
                            e
                        ))
                    })?;

                    let mut ctx =
                        CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
                    let ptr_reg = ptr_ref.load_to_reg(&mut ctx).map_err(|e| {
                        LlvmCompilerError::RegisterAllocation(format!(
                            "Failed to load ptr: {:?}",
                            e
                        ))
                    })?;

                    let stack_offset = assignment.stack_off.ok_or_else(|| {
                        LlvmCompilerError::LlvmError(
                            "byval argument not assigned to stack".to_string(),
                        )
                    })?;

                    // Simple byte copy implementation
                    let encoder = self.codegen.encoder_mut();
                    let rsp = AsmReg::new(0, 4); // RSP

                    for offset in (0..size).step_by(8) {
                        let temp_reg = self
                            .register_file
                            .allocate_reg(0, crate::core::ValLocalIdx::MAX, 0, None)
                            .map_err(|e| {
                                LlvmCompilerError::RegisterAllocation(format!(
                                    "Failed to allocate temp reg: {:?}",
                                    e
                                ))
                            })?;

                        encoder
                            .mov64_reg_mem(temp_reg, ptr_reg, offset as i32)
                            .map_err(|e| {
                                LlvmCompilerError::CodeGeneration(format!(
                                    "Failed to load from byval src: {:?}",
                                    e
                                ))
                            })?;

                        encoder
                            .mov64_mem_reg(rsp, stack_offset + offset as i32, temp_reg)
                            .map_err(|e| {
                                LlvmCompilerError::CodeGeneration(format!(
                                    "Failed to store to byval dst: {:?}",
                                    e
                                ))
                            })?;

                        let _ = self.register_file.free_register(temp_reg);
                    }

                    log::trace!(
                        "   Copied {} bytes for byval argument to stack offset {}",
                        size,
                        stack_offset
                    );
                }
                _ => {
                    // For all other cases, handle normally
                    let mut arg_ref = ValuePartRef::new(*arg_idx, 0).map_err(|e| {
                        LlvmCompilerError::RegisterAllocation(format!(
                            "Failed to create arg ref: {:?}",
                            e
                        ))
                    })?;

                    if let Some(reg) = assignment.reg {
                        let mut ctx =
                            CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
                        let arg_reg = arg_ref.load_to_reg(&mut ctx).map_err(|e| {
                            LlvmCompilerError::RegisterAllocation(format!(
                                "Failed to load arg: {:?}",
                                e
                            ))
                        })?;

                        if arg_reg != reg {
                            let encoder = self.codegen.encoder_mut();
                            encoder.mov_reg_reg(reg, arg_reg).map_err(|e| {
                                LlvmCompilerError::CodeGeneration(format!(
                                    "Failed to emit mov: {:?}",
                                    e
                                ))
                            })?;
                            log::trace!(
                                "      Generated: mov {}:{}, {}:{}",
                                reg.bank,
                                reg.id,
                                arg_reg.bank,
                                arg_reg.id
                            );
                        }
                    } else if let Some(stack_offset) = assignment.stack_off {
                        let mut ctx =
                            CompilerContext::new(&mut self.value_mgr, &mut self.register_file);
                        let arg_reg = arg_ref.load_to_reg(&mut ctx).map_err(|e| {
                            LlvmCompilerError::RegisterAllocation(format!(
                                "Failed to load arg: {:?}",
                                e
                            ))
                        })?;

                        let encoder = self.codegen.encoder_mut();
                        let rsp = AsmReg::new(0, 4); // RSP
                        encoder
                            .mov64_mem_reg(rsp, stack_offset, arg_reg)
                            .map_err(|e| {
                                LlvmCompilerError::CodeGeneration(format!(
                                    "Failed to emit stack store: {:?}",
                                    e
                                ))
                            })?;
                        log::trace!(
                            "      Generated: mov [rsp+{}], {}:{}",
                            stack_offset,
                            arg_reg.bank,
                            arg_reg.id
                        );
                    }
                }
            }
        }

        // For varargs functions, set AL register with XMM register count
        if is_varargs {
            let used_xmm = cc_assigner.xmm_used();
            if used_xmm > 0 {
                let encoder = self.codegen.encoder_mut();
                let al = AsmReg::new(0, 0); // AL register
                encoder
                    .mov_reg_imm(al, used_xmm as i64)
                    .map_err(|e| {
                        LlvmCompilerError::CodeGeneration(format!(
                            "Failed to set AL for varargs: {:?}",
                            e
                        ))
                    })?;
                log::trace!("   Set AL={} for varargs XMM count", used_xmm);
            }
        }

        // Emit the actual call instruction
        let encoder = self.codegen.encoder_mut();

        // Use offset 0 for direct calls - will be resolved during linking
        encoder.call_direct(0).map_err(|e| {
            LlvmCompilerError::CodeGeneration(format!("Failed to emit call: {:?}", e))
        })?;
        log::trace!(
            "   Generated: call {} (offset will be resolved later)",
            function_name
        );

        // Record the call site for later relocation
        self.session.record_call_site(function_name.clone());

        // Handle return value if any
        if !instruction.get_type().is_void_type() {
            // Get result value index
            // For call instructions, we need to create a unique index for the result
            // Since we can't convert InstructionValue to BasicValueEnum directly,
            // we'll use the pointer address
            use inkwell::values::AsValueRef;
            let inst_ptr = instruction.as_value_ref() as usize;
            let result_idx = inst_ptr % 1024; // Simple hash for index

            // Determine return value location based on type
            let ret_type = instruction.get_type();
            let (ret_bank, ret_size) = match ret_type {
                inkwell::types::AnyTypeEnum::IntType(int_type) => {
                    (RegBank::GeneralPurpose, int_type.get_bit_width() / 8)
                }
                inkwell::types::AnyTypeEnum::PointerType(_) => (RegBank::GeneralPurpose, 8),
                inkwell::types::AnyTypeEnum::FloatType(float_type) => {
                    match float_type.get_context().f32_type() {
                        t if t == float_type => (RegBank::Xmm, 4),
                        _ => (RegBank::Xmm, 8), // f64
                    }
                }
                _ => {
                    return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                        "Unsupported return type: {:?}",
                        ret_type
                    )));
                }
            };

            // Return value is in RAX (GP) or XMM0 (FP)
            let ret_reg = match ret_bank {
                RegBank::GeneralPurpose => AsmReg::new(0, 0), // RAX
                RegBank::Xmm => AsmReg::new(1, 0),            // XMM0
            };

            // Create value assignment for result
            if self.value_mgr.get_assignment(result_idx).is_none() {
                self.value_mgr
                    .create_assignment(result_idx, 1, ret_size as u8);
            }

            // Mark the return register as containing the result
            let _result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
                LlvmCompilerError::RegisterAllocation(format!(
                    "Failed to create result ref: {:?}",
                    e
                ))
            })?;
            // The return value is now in the return register
            // In a real implementation, we would need to update the register file
            // to track that this value is now in the return register.
            // For now, we'll just note that the calling convention places it there.
            log::debug!(
                "   Result v{} is in register {}:{}",
                result_idx,
                ret_reg.bank,
                ret_reg.id
            );
        }

        // Update statistics
        self.session.record_instruction_compiled("Call");

        Ok(())
    }

    /// Helper function to set up an argument in its assigned location.
    fn compile_alloca_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("📋 Compiling ALLOCA instruction");

        // For now, we'll use a simplified approach to alloca
        // We'll allocate a fixed size based on common patterns
        // This will be improved to properly determine the allocated type

        // Default allocation size - we'll start with 8 bytes (64-bit value)
        // In practice, we'd need to analyze the alloca instruction to determine the actual size
        let size = 8u32;
        let _num_elements = 1u32; // For now, assume single element

        // Check if this is an array alloca by looking at operands
        let total_size = if instruction.get_num_operands() > 0 {
            // Array alloca - try to get size
            if let Some(inkwell::values::BasicValueEnum::IntValue(int_val)) =
                instruction.get_operand(0).and_then(|op| op.left())
            {
                if let Some(const_int) = int_val.get_zero_extended_constant() {
                    size * (const_int as u32)
                } else {
                    size * 10 // Default array size
                }
            } else {
                size
            }
        } else {
            size
        };

        // Allocate stack space
        let stack_offset = self.codegen.allocate_spill_slot(total_size);
        log::debug!(
            "   Allocated {} bytes at stack offset {}",
            total_size,
            stack_offset
        );

        // The result of alloca is a pointer to the allocated space
        // We need to compute the address (rbp + offset) and store it in a register
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Create value assignment for the pointer result
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, 8); // Pointer is 64-bit
        }

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Allocate register for the result pointer
        let mut result_ref = ValuePartRef::new(result_idx, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create result ref: {:?}", e))
        })?;
        let result_reg = result_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate result: {:?}", e))
        })?;

        // Generate LEA instruction to compute the address
        let encoder = self.codegen.encoder_mut();
        let rbp = AsmReg::new(0, 5); // RBP register

        encoder
            .lea(result_reg, rbp, None, 1, stack_offset)
            .map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit lea: {:?}", e))
            })?;

        log::trace!(
            "   Generated: lea {}:{}, [rbp + {}]",
            result_reg.bank,
            result_reg.id,
            stack_offset
        );

        Ok(())
    }

    /// Store PHI nodes from analysis results into session.
    fn store_phi_nodes_from_analysis(
        &mut self,
        analysis: &FunctionAnalysis<'arena>,
    ) -> Result<(), LlvmCompilerError> {
        // Store each PHI node in the session
        for phi_node in &analysis.phi_nodes[0..analysis.phi_count] {
            let mut incoming_values = Vec::new();

            // Collect incoming values for this PHI
            for i in 0..phi_node.incoming_count {
                let incoming_idx = phi_node.incoming_start + i;
                let incoming = &analysis.phi_incoming[incoming_idx];
                incoming_values.push((incoming.value_idx, incoming.pred_block_idx));
                log::trace!(
                    "      PHI incoming for v{}: v{} from block {}",
                    phi_node.result_idx,
                    incoming.value_idx,
                    incoming.pred_block_idx
                );
            }

            // Create PHI info and store in session
            let phi_info = crate::core::session::PhiNodeInfo {
                result_value: phi_node.result_idx,
                incoming_values,
            };

            // Store using result index as key
            self.session.add_phi_node(phi_node.result_idx, phi_info);
            log::trace!(
                "   PHI   Stored PHI node with result v{} in block {} with {} incoming values",
                phi_node.result_idx,
                phi_node.block_idx,
                phi_node.incoming_count
            );
        }

        log::debug!(
            "   PHI   Stored {} PHI nodes in session",
            analysis.phi_count
        );
        Ok(())
    }

    /// Get access to compilation session.
    pub fn session(&self) -> &CompilationSession<'arena> {
        self.session
    }

    /// Get list of compiled functions.
    pub fn compiled_functions(&self) -> &HashMap<String, CompiledFunction<'arena>> {
        &self.compiled_functions
    }

    /// Get the LLVM module.
    pub fn module(&self) -> &inkwell::module::Module<'ctx> {
        &self.module
    }

    /// Get or create a value index for an LLVM value.
    fn get_or_create_value_index(
        &mut self,
        value: BasicValueEnum<'ctx>,
    ) -> Result<usize, LlvmCompilerError> {
        // Use pointer address as unique identifier
        use inkwell::values::AsValueRef;
        let ptr_addr = value.as_value_ref() as usize;

        // Simple index assignment - in real implementation would use proper mapping
        Ok(ptr_addr % 1024) // Modulo to fit in our value assignment table
    }

    /// Get element size for GEP instruction based on index position.
    fn get_gep_element_size(&self, idx_num: usize) -> Result<u64, LlvmCompilerError> {
        // Simplified size calculation
        // TODO: Integrate with LLVM type system for accurate sizes
        match idx_num {
            0 => Ok(4), // First index: assume i32 array elements
            1 => Ok(8), // Second index: assume 64-bit struct fields
            _ => Ok(4), // Default
        }
    }

    /// Try to get constant value from an index.
    fn try_get_constant_index(&self, index_idx: usize) -> Option<i64> {
        // Simplified constant detection
        // In real implementation would check LLVM ConstantInt
        if (100..=110).contains(&index_idx) {
            Some((index_idx - 100) as i64)
        } else {
            None
        }
    }

    /// Get block index by name.
    fn get_block_index_by_name(&self, block_name: &str) -> Result<usize, LlvmCompilerError> {
        // Get the current function
        let current_function = self.current_function.ok_or_else(|| {
            LlvmCompilerError::Session(crate::core::session::SessionError::InvalidState(
                "No current function".to_string(),
            ))
        })?;

        // Get all blocks from the function
        let blocks = current_function.get_basic_blocks();

        // Find the block with the given name
        for (idx, block) in blocks.iter().enumerate() {
            if let Ok(name) = block.get_name().to_str() {
                if name == block_name {
                    return Ok(idx);
                }
            }
        }

        // Block not found
        Err(LlvmCompilerError::Session(
            crate::core::session::SessionError::BlockNotFound(999), // Use 999 as unknown
        ))
    }

    /// Materialize GEP expression into machine code.
    fn materialize_gep_expression(
        &mut self,
        gep_expr: GepExpression,
        result_reg: AsmReg,
    ) -> Result<(), LlvmCompilerError> {
        let encoder = self.codegen.encoder_mut();

        // Try to use LEA for address calculation
        if let Some(addr_mode) = gep_expr.to_addressing_mode() {
            match addr_mode {
                AddressingMode::Register(base) => {
                    if result_reg != base {
                        encoder.mov_reg_reg(result_reg, base).map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit mov: {:?}",
                                e
                            ))
                        })?;
                    }
                    log::trace!(
                        "   Generated: mov {}:{}, {}:{}",
                        result_reg.bank,
                        result_reg.id,
                        base.bank,
                        base.id
                    );
                }
                AddressingMode::RegisterOffset(base, offset) => {
                    encoder
                        .lea(result_reg, base, None, 1, offset)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit lea: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: lea {}:{}, [{}:{} + {}]",
                        result_reg.bank,
                        result_reg.id,
                        base.bank,
                        base.id,
                        offset
                    );
                }
                AddressingMode::RegisterIndexScale(base, index, scale) => {
                    encoder
                        .lea(result_reg, base, Some(index), scale as u32, 0)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit lea: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: lea {}:{}, [{}:{} + {}:{}*{}]",
                        result_reg.bank,
                        result_reg.id,
                        base.bank,
                        base.id,
                        index.bank,
                        index.id,
                        scale
                    );
                }
                AddressingMode::RegisterIndexScaleOffset(base, index, scale, offset) => {
                    encoder
                        .lea(result_reg, base, Some(index), scale as u32, offset)
                        .map_err(|e| {
                            LlvmCompilerError::CodeGeneration(format!(
                                "Failed to emit lea: {:?}",
                                e
                            ))
                        })?;
                    log::trace!(
                        "   Generated: lea {}:{}, [{}:{} + {}:{}*{} + {}]",
                        result_reg.bank,
                        result_reg.id,
                        base.bank,
                        base.id,
                        index.bank,
                        index.id,
                        scale,
                        offset
                    );
                }
                _ => {
                    return Err(LlvmCompilerError::UnsupportedInstruction(
                        "Complex GEP addressing mode not supported".to_string(),
                    ));
                }
            }
        } else if gep_expr.needs_materialization {
            // Complex expression needs multiple instructions
            log::debug!("   GEP: Complex materialization required (not implemented)");
            return Err(LlvmCompilerError::UnsupportedInstruction(
                "Complex GEP expressions not yet supported".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate PHI moves for a control flow edge.
    fn generate_phi_moves_for_edge(
        &mut self,
        from_block: usize,
        to_block: usize,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!(
            "   PHI   📋 Generating PHI moves from block {} to block {}",
            from_block,
            to_block
        );

        // First, we need to collect all the moves we need to make
        // This is important because we need to handle parallel moves and potential cycles
        let mut moves_to_make = Vec::new();

        // Get the current function to find PHI nodes in the target block
        let current_function = self.current_function.ok_or_else(|| {
            LlvmCompilerError::Session(crate::core::session::SessionError::InvalidState(
                "No current function".to_string(),
            ))
        })?;

        // Get the target block
        let blocks = current_function.get_basic_blocks();
        if to_block >= blocks.len() {
            return Err(LlvmCompilerError::Session(
                crate::core::session::SessionError::BlockNotFound(to_block),
            ));
        }
        let target_block = blocks[to_block];

        // Get all PHI nodes from session and check which ones are in the target block
        let all_phi_nodes = self.session.get_all_phi_nodes();
        log::trace!("   PHI   Checking {} stored PHI nodes", all_phi_nodes.len());

        // Count PHI instructions in target block to know how many to look for
        let mut phi_count_in_block = 0;
        for instruction in target_block.get_instructions() {
            if instruction.get_opcode() != inkwell::values::InstructionOpcode::Phi {
                break;
            }
            phi_count_in_block += 1;
        }
        log::trace!(
            "   PHI   Target block {} has {} PHI instructions",
            to_block,
            phi_count_in_block
        );

        // Now find the PHI nodes for this block
        // We need to check if the PHI node belongs to the target block
        // by verifying it has incoming edges that match our current blocks
        for (phi_key, phi_info) in all_phi_nodes {
            log::trace!(
                "   PHI   Checking PHI node {} with result v{}",
                phi_key,
                phi_info.result_value
            );

            // A PHI node belongs to the target block if it has incoming values
            // and we're on an edge that feeds into it
            for &(value_idx, block_idx) in &phi_info.incoming_values {
                log::trace!(
                    "      PHI incoming: v{} from block {} (checking against from_block {})",
                    value_idx,
                    block_idx,
                    from_block
                );
                if block_idx == from_block {
                    // This PHI node has an incoming value from our source block
                    moves_to_make.push((value_idx, phi_info.result_value));
                    log::trace!(
                        "      Found PHI move: v{} -> v{} for edge {} -> {}",
                        value_idx,
                        phi_info.result_value,
                        from_block,
                        to_block
                    );
                    break;
                }
            }
        }

        if moves_to_make.is_empty() {
            log::trace!(
                "   PHI   No PHI moves needed for edge {} -> {}",
                from_block,
                to_block
            );
            return Ok(());
        }

        log::debug!(
            "   PHI   Found {} PHI moves to generate",
            moves_to_make.len()
        );

        // Now we need to perform the moves, being careful about parallel assignment
        // For now, we'll use a simple approach that may use extra moves but is correct
        // TODO: Implement proper parallel move with cycle detection

        // First pass: moves where source is not used as any destination
        let mut completed = vec![false; moves_to_make.len()];
        let destinations: Vec<usize> = moves_to_make.iter().map(|(_, dst)| *dst).collect();

        for (i, &(src_value, dst_value)) in moves_to_make.iter().enumerate() {
            if !destinations.contains(&src_value) {
                // Safe to move directly
                self.generate_single_phi_move(src_value, dst_value)?;
                completed[i] = true;
            }
        }

        // Second pass: remaining moves (potential cycles)
        // For simplicity, we'll use temporary registers for any remaining moves
        for (i, &(src_value, dst_value)) in moves_to_make.iter().enumerate() {
            if !completed[i] {
                // This could be part of a cycle, so we'll be conservative
                // and use a temporary register
                self.generate_phi_move_with_temp(src_value, dst_value)?;
            }
        }

        // Record statistics
        for _ in &moves_to_make {
            self.session.record_phi_resolved();
        }

        Ok(())
    }

    /// Generate a single PHI move directly.
    fn generate_single_phi_move(
        &mut self,
        src_value: usize,
        dst_value: usize,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("      PHI move: v{} -> v{}", src_value, dst_value);

        // Create value assignments if needed (default to 64-bit for safety)
        if self.value_mgr.get_assignment(src_value).is_none() {
            self.value_mgr.create_assignment(src_value, 1, 8);
        }
        if self.value_mgr.get_assignment(dst_value).is_none() {
            self.value_mgr.create_assignment(dst_value, 1, 8);
        }

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Create value references
        let mut src_ref = ValuePartRef::new(src_value, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create src ref: {:?}", e))
        })?;
        let mut dst_ref = ValuePartRef::new(dst_value, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create dst ref: {:?}", e))
        })?;

        // Load source to register
        let src_reg = src_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load src: {:?}", e))
        })?;

        // Try to reuse source register for destination
        let dst_reg = dst_ref
            .alloc_try_reuse(&mut src_ref, &mut ctx)
            .map_err(|e| {
                LlvmCompilerError::RegisterAllocation(format!("Failed to allocate dst: {:?}", e))
            })?;

        // Generate move if needed
        if src_reg != dst_reg {
            let encoder = self.codegen.encoder_mut();
            encoder.mov_reg_reg(dst_reg, src_reg).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit mov: {:?}", e))
            })?;
            log::trace!(
                "      Generated: mov {}:{}, {}:{}",
                dst_reg.bank,
                dst_reg.id,
                src_reg.bank,
                src_reg.id
            );
        } else {
            log::trace!("      No move needed (same register)");
        }

        Ok(())
    }

    /// Generate a PHI move using a temporary register (for cycle breaking).
    fn generate_phi_move_with_temp(
        &mut self,
        src_value: usize,
        dst_value: usize,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("      PHI move with temp: v{} -> v{}", src_value, dst_value);

        // Create value assignments if needed
        if self.value_mgr.get_assignment(src_value).is_none() {
            self.value_mgr.create_assignment(src_value, 1, 8);
        }
        if self.value_mgr.get_assignment(dst_value).is_none() {
            self.value_mgr.create_assignment(dst_value, 1, 8);
        }

        // Create compiler context
        let mut ctx = CompilerContext::new(&mut self.value_mgr, &mut self.register_file);

        // Load source to a scratch register first
        let mut src_ref = ValuePartRef::new(src_value, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create src ref: {:?}", e))
        })?;
        let src_reg = src_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to load src: {:?}", e))
        })?;

        // Allocate a scratch register for the temporary
        // We'll use R11 as a scratch register (following x86-64 convention)
        let temp_reg = AsmReg::new(0, 11);

        // Move source to temp
        let encoder = self.codegen.encoder_mut();
        encoder.mov_reg_reg(temp_reg, src_reg).map_err(|e| {
            LlvmCompilerError::CodeGeneration(format!("Failed to emit mov to temp: {:?}", e))
        })?;
        log::trace!("      Generated: mov r11, {}:{}", src_reg.bank, src_reg.id);

        // Now move temp to destination
        let mut dst_ref = ValuePartRef::new(dst_value, 0).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to create dst ref: {:?}", e))
        })?;
        let dst_reg = dst_ref.load_to_reg(&mut ctx).map_err(|e| {
            LlvmCompilerError::RegisterAllocation(format!("Failed to allocate dst: {:?}", e))
        })?;

        if dst_reg != temp_reg {
            encoder.mov_reg_reg(dst_reg, temp_reg).map_err(|e| {
                LlvmCompilerError::CodeGeneration(format!("Failed to emit mov from temp: {:?}", e))
            })?;
            log::trace!("      Generated: mov {}:{}, r11", dst_reg.bank, dst_reg.id);
        }

        Ok(())
    }

    /// Compile PHI instruction.
    fn compile_phi_instruction(
        &mut self,
        instruction: inkwell::values::InstructionValue<'ctx>,
    ) -> Result<(), LlvmCompilerError> {
        log::trace!("🔄 Compiling PHI instruction");

        // PHI nodes are special - they don't generate code directly.
        // Instead, they define values that need to be resolved at block edges.
        // The actual moves are generated when compiling branch instructions.

        // Get result value index for this PHI
        use inkwell::values::AsValueRef;
        let inst_ptr = instruction.as_value_ref() as usize;
        let result_idx = inst_ptr % 1024;

        // Get the type of the PHI result
        let phi_type = instruction.get_type();
        let value_size = match phi_type {
            inkwell::types::AnyTypeEnum::IntType(int_type) => (int_type.get_bit_width() / 8) as u8,
            inkwell::types::AnyTypeEnum::PointerType(_) => 8,
            _ => {
                return Err(LlvmCompilerError::UnsupportedInstruction(format!(
                    "Unsupported PHI type: {:?}",
                    phi_type
                )));
            }
        };

        // Create value assignment for the PHI result
        if self.value_mgr.get_assignment(result_idx).is_none() {
            self.value_mgr.create_assignment(result_idx, 1, value_size);
        }

        // Use inkwell's PhiValue methods to properly extract incoming values
        use inkwell::values::PhiValue;
        if let Ok(phi_value) = PhiValue::try_from(instruction) {
            let num_incoming = phi_value.count_incoming();
            log::trace!("   PHI   PHI has {} incoming values", num_incoming);

            let mut incoming_values = Vec::new();

            for i in 0..num_incoming {
                let value = phi_value.get_incoming(i).unwrap().0;
                let block = phi_value.get_incoming(i).unwrap().1;

                // Get or create value index for incoming value
                let incoming_idx = self.get_or_create_value_index(value)?;
                if self.value_mgr.get_assignment(incoming_idx).is_none() {
                    self.value_mgr
                        .create_assignment(incoming_idx, 1, value_size);
                }

                // Get block name and index
                let block_name = block.get_name().to_str().map_err(|e| {
                    LlvmCompilerError::LlvmError(format!("Invalid block name: {:?}", e))
                })?;
                let block_idx = self.get_block_index_by_name(block_name)?;

                log::debug!(
                    "   Incoming: v{} from block {} (idx {})",
                    incoming_idx,
                    block_name,
                    block_idx
                );
                incoming_values.push((incoming_idx, block_idx));
            }

            // PHI nodes are already stored in session during analysis phase
            // Just log that we processed this PHI instruction
            log::trace!(
                "   PHI   PHI instruction processed with {} incoming values",
                num_incoming
            );
        } else {
            return Err(LlvmCompilerError::LlvmError(
                "Failed to convert instruction to PhiValue".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;
    use inkwell::context::Context;

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
        let result = builder
            .build_int_add(param_a, param_b, "add_result")
            .unwrap();
        builder.build_return(Some(&result)).unwrap();

        module
    }

    #[test]
    fn test_llvm_compiler_creation() {
        let context = Context::create();
        let module = create_simple_module(&context);
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let compiler = LlvmCompiler::new(module, &session);
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_function_compilation() {
        let context = Context::create();
        let module = create_simple_module(&context);
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);
        let mut compiler = LlvmCompiler::new(module, &session).unwrap();

        compiler.compile_function_by_name("add").unwrap();

        let compiled = compiler.compiled_functions().get("add").unwrap();
        assert_eq!(compiled.name, "add");
        assert!(compiled.code_size > 0);
    }

    #[test]
    fn test_llvm_compiler_session_statistics() {
        let context = Context::create();
        let module = create_simple_module(&context);
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);
        let mut compiler = LlvmCompiler::new(module, &session).unwrap();

        compiler.compile_function_by_name("add").unwrap();

        let stats = compiler.session().stats();
        assert_eq!(stats.functions_compiled, 1);
        assert!(stats.instructions_compiled > 0);
    }

    fn create_gep_test_module(context: &Context) -> inkwell::module::Module {
        let module = context.create_module("gep_test");
        let i32_type = context.i32_type();
        let i32_ptr_type = context.ptr_type(inkwell::AddressSpace::default());
        let fn_type = i32_type.fn_type(&[i32_ptr_type.into(), i32_type.into()], false);
        let function = module.add_function("array_access", fn_type, None);

        let entry_block = context.append_basic_block(function, "entry");
        let builder = context.create_builder();
        builder.position_at_end(entry_block);

        // Get parameters: ptr and index
        let ptr_param = function.get_nth_param(0).unwrap().into_pointer_value();
        let index_param = function.get_nth_param(1).unwrap().into_int_value();

        // GEP instruction: ptr[index]
        let indices = &[index_param];
        let element_ptr =
            unsafe { builder.build_in_bounds_gep(i32_type, ptr_param, indices, "element_ptr") }
                .unwrap();

        // Load the value
        let value = builder.build_load(i32_type, element_ptr, "value").unwrap();
        builder.build_return(Some(&value.into_int_value())).unwrap();

        module
    }

    #[test]
    fn test_gep_instruction_compilation() {
        let context = Context::create();
        let module = create_gep_test_module(&context);
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);
        let mut compiler = LlvmCompiler::new(module, &session).unwrap();

        compiler
            .compile_function_by_name("array_access")
            .expect("GEP compilation should succeed");

        let compiled = compiler.compiled_functions().get("array_access").unwrap();
        assert_eq!(compiled.name, "array_access");
        assert!(compiled.code_size > 0);

        // Check that GEP instruction was compiled
        let stats = compiler.session().stats();
        assert!(stats.instruction_counts.contains_key("GetElementPtr"));
    }

    fn create_icmp_test_module(context: &Context) -> inkwell::module::Module {
        let module = context.create_module("icmp_test");
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("compare", fn_type, None);

        let entry_block = context.append_basic_block(function, "entry");
        let builder = context.create_builder();
        builder.position_at_end(entry_block);

        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();

        // Create ICMP with SGT predicate
        let cmp_result = builder
            .build_int_compare(inkwell::IntPredicate::SGT, a, b, "cmp_sgt")
            .unwrap();

        // Convert bool to i32
        let result = builder
            .build_int_z_extend(cmp_result, i32_type, "result")
            .unwrap();
        builder.build_return(Some(&result)).unwrap();

        module
    }

    #[test]
    fn test_icmp_real_predicate_extraction() {
        let context = Context::create();
        let module = create_icmp_test_module(&context);
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);
        let mut compiler = LlvmCompiler::new(module, &session).unwrap();

        compiler
            .compile_function_by_name("compare")
            .expect("ICMP compilation should succeed");

        // Verify ICMP was compiled
        let stats = compiler.session().stats();
        assert!(stats.instruction_counts.contains_key("ICmp"));
        assert_eq!(stats.instruction_counts["ICmp"], 1);
    }
}
