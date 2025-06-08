# Elegant Code Generation for TPDE Rust - Implementation Guide

## CRITICAL: This Document is the Implementation Specification

**This document MUST be followed EXACTLY when implementing the v2 architecture. Every section contains MANDATORY requirements that MUST be implemented as specified.**

## Overview

The Rust TPDE implementation MUST be transformed from manual instruction encoding to an elegant template-based approach. The current implementation is UNACCEPTABLE due to its verbosity and maintenance burden. This transformation is NOT optional.

## Current State Analysis

### Problem - MUST READ AND UNDERSTAND
The current Rust implementation is FAILING due to:
- **THOUSANDS of lines of manual encoding** - See `src/x64/encoder.rs` lines 1335-1582 for floating point alone
- **EXTREME code duplication** - See `src/llvm/compiler.rs` lines 2800-3300 for repetitive patterns
- **ERROR-PRONE manual register management** - Every instruction requires 20+ lines of boilerplate
- **MISSED optimizations** - No instruction fusion, no immediate folding

**MANDATORY FILES TO READ BEFORE IMPLEMENTATION:**
1. `src/x64/encoder.rs` - Contains ALL manual instruction encodings that MUST be replaced
2. `src/llvm/compiler.rs` - Contains ALL manual compilation functions that MUST be simplified
3. `src/core/value_ref.rs` - ValueRef and ScratchReg types that MUST be used in generated code
4. `src/core/register_file.rs` - RegisterFile and AsmReg types used throughout

### C++ Solution - STUDY THESE FILES
The C++ version demonstrates the CORRECT approach:
1. **READ** `tpde-llvm/src/encode_template.c` - Simple C functions that define ALL operations
2. **READ** `tpde-encodegen/src/main.cpp` - How templates are processed
3. **READ** `tpde-encodegen/src/x64/Target.cpp` - How patterns are extracted
4. **READ** `docs/tpde/encodegen.md` - Full documentation of the C++ approach

## Proposed Rust Solution

### Core Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Template Funcs  │────►│ build.rs     │────►│ Generated Code  │
│ (Rust code)     │     │ (Extraction) │     │ (encoders.rs)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                                              │
         │                                              ▼
         │              ┌──────────────┐      ┌─────────────────┐
         └─────────────►│ Proc Macros  │─────►│ Compiler Impl   │
                        │ (Analysis)   │      │ (llvm/compiler) │
                        └──────────────┘      └─────────────────┘
```

### 1. Template Definition Layer - MANDATORY IMPLEMENTATION

**YOU MUST CREATE** the file `src/templates/mod.rs` with EXACTLY this structure:

```rust
// src/templates/mod.rs
// THIS FILE IS MANDATORY - IT DEFINES ALL INSTRUCTION TEMPLATES

#![allow(dead_code)]
#![cfg_attr(feature = "template-extraction", no_std)]

use core::arch::x86_64::*;

// MANDATORY: Support ALL x86_64 microarchitecture levels (v1, v2, v3, v4)
// Each template MUST be annotated with the minimum required level

// x86_64-v1: Baseline (SSE, SSE2)
#[template_function(target = "x86_64-v1")]
pub fn add_f32(a: f32, b: f32) -> f32 { a + b }

#[template_function(target = "x86_64-v1")]
pub fn add_f64(a: f64, b: f64) -> f64 { a + b }

#[template_function(target = "x86_64-v1")]
pub fn sub_f32(a: f32, b: f32) -> f32 { a - b }

#[template_function(target = "x86_64-v1")]
pub fn sub_f64(a: f64, b: f64) -> f64 { a - b }

#[template_function(target = "x86_64-v1")]
pub fn mul_f32(a: f32, b: f32) -> f32 { a * b }

#[template_function(target = "x86_64-v1")]
pub fn mul_f64(a: f64, b: f64) -> f64 { a * b }

#[template_function(target = "x86_64-v1")]
pub fn div_f32(a: f32, b: f32) -> f32 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn div_f64(a: f64, b: f64) -> f64 { a / b }

// MANDATORY: Define ALL integer operations from encode_template.c lines 136-165
#[template_function(target = "x86_64-v1")]
pub fn add_i32(a: i32, b: i32) -> i32 { a.wrapping_add(b) }

#[template_function(target = "x86_64-v1")]
pub fn add_i64(a: i64, b: i64) -> i64 { a.wrapping_add(b) }

// MANDATORY: Complex patterns that MUST be recognized
#[template_function(target = "x86_64-v1", inline_mode = "always")]
pub fn select_i32(cond: u8, a: i32, b: i32) -> i32 {
    if cond & 1 != 0 { a } else { b }
}

// MANDATORY: Memory operations with addressing modes
#[template_function(target = "x86_64-v1")]
pub unsafe fn load_i32(ptr: *const i32) -> i32 { *ptr }

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_i32(ptr: *mut i32, value: i32) { *ptr = value; }

// x86_64-v2: Add SSSE3, SSE4.1, SSE4.2, POPCNT
#[template_function(target = "x86_64-v2")]
pub fn popcnt_i32(a: i32) -> i32 {
    a.count_ones() as i32
}

#[template_function(target = "x86_64-v2")]
pub fn popcnt_i64(a: i64) -> i32 {
    a.count_ones() as i32
}

// x86_64-v3: Add AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE
#[template_function(target = "x86_64-v3")]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    // Rust will optimize this to FMA instruction with target-cpu=x86-64-v3
    a.mul_add(b, c)
}

#[template_function(target = "x86_64-v3")]
pub fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[template_function(target = "x86_64-v3")]
pub fn lzcnt_i32(a: i32) -> i32 {
    a.leading_zeros() as i32
}

#[template_function(target = "x86_64-v3")]
pub fn lzcnt_i64(a: i64) -> i32 {
    a.leading_zeros() as i32
}

// x86_64-v4: Add AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL
#[template_function(target = "x86_64-v4")]
pub fn add_f32x16(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut result = [0.0f32; 16];
    for i in 0..16 {
        result[i] = a[i] + b[i];
    }
    result
}

// YOU MUST IMPLEMENT ALL OTHER OPERATIONS FROM encode_template.c
// WITH APPROPRIATE TARGET ANNOTATIONS FOR EACH MICROARCHITECTURE LEVEL
```

### 2. Build-Time Extraction (build.rs) - CRITICAL IMPLEMENTATION

**YOU MUST REPLACE** the existing `build.rs` with this EXACT implementation:

```rust
// build.rs
// THIS IS MANDATORY - THIS DRIVES THE ENTIRE CODE GENERATION

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // MANDATORY: Rerun if templates change
    println!("cargo:rerun-if-changed=src/templates/");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let target = env::var("TARGET").unwrap();
    
    // MANDATORY STEP 1: Compile templates to LLVM IR
    let ir_path = compile_templates_to_ir(&out_dir, &target);
    
    // MANDATORY STEP 2: Run LLVM optimization passes
    let opt_ir_path = optimize_ir(&ir_path, &out_dir);
    
    // MANDATORY STEP 3: Extract machine code patterns using LLVM
    let patterns = extract_instruction_patterns(&opt_ir_path, &target);
    
    // MANDATORY STEP 4: Generate Rust encoder implementations
    generate_encoder_impls(&patterns, &out_dir);
    
    // MANDATORY STEP 5: Generate pattern matching tables
    generate_pattern_tables(&patterns, &out_dir);
    
    // MANDATORY STEP 6: Generate integration code
    generate_compiler_integration(&patterns, &out_dir);
}

fn compile_templates_to_ir(out_dir: &str, target: &str) -> PathBuf {
    let ir_path = Path::new(out_dir).join("templates.ll");
    
    // YOU MUST use these EXACT flags for correct code generation
    // MANDATORY: Compile templates for each microarchitecture level
    let cpu_targets = match target {
        "x86_64-v1" => "x86-64",
        "x86_64-v2" => "x86-64-v2",
        "x86_64-v3" => "x86-64-v3", 
        "x86_64-v4" => "x86-64-v4",
        _ => "x86-64", // Default to baseline
    };
    
    let status = Command::new("rustc")
        .args(&[
            "src/templates/mod.rs",
            "--crate-type=rlib",
            "--emit=llvm-ir",
            "-C", "opt-level=3",
            "-C", &format!("target-cpu={}", cpu_targets),
            "-C", "no-redzone=yes",
            "-C", "panic=abort",
            "--cfg", "feature=\"template-extraction\"",
            "--target", target,
            "-o", ir_path.to_str().unwrap(),
        ])
        .status()
        .expect("FATAL: Failed to compile templates to LLVM IR");
    
    assert!(status.success(), "FATAL: Template compilation failed");
    ir_path
}

// YOU MUST IMPLEMENT: Use llvm-sys or inkwell to parse IR and extract patterns
fn extract_instruction_patterns(ir_path: &Path, target: &str) -> Vec<InstructionPattern> {
    // MANDATORY: Parse LLVM IR using inkwell
    // MANDATORY: Extract function bodies
    // MANDATORY: Run LLVM codegen to get machine instructions
    // MANDATORY: Pattern match on instruction sequences
    
    // TEMPORARY: You MUST replace this with real implementation
    panic!("YOU MUST IMPLEMENT extract_instruction_patterns using inkwell/llvm-sys");
}

#[derive(Debug, Clone)]
struct InstructionPattern {
    pub name: String,
    pub rust_signature: String,
    pub llvm_signature: String,
    pub machine_code: Vec<u8>,
    pub instruction_sequence: Vec<MachineInst>,
    pub can_fold_immediate: bool,
    pub can_fold_memory: bool,
}

// YOU MUST generate code that integrates with existing types
fn generate_encoder_impls(patterns: &[InstructionPattern], out_dir: &str) {
    let mut code = String::from(
        "// GENERATED CODE - DO NOT EDIT
use crate::core::{ValuePartRef, ScratchReg, CompilerContext, AsmReg};
use crate::x64::encoder::{X64Encoder, EncodingError};

"
    );
    
    // Generate encoder for each pattern
    for pattern in patterns {
        code.push_str(&format!(
            "pub fn encode_{}(
    encoder: &mut X64Encoder,
    ctx: &mut CompilerContext,
    inputs: &[ValuePartRef],
    output: &mut ScratchReg,
) -> Result<(), EncodingError> {{
    // GENERATED: Pattern-specific encoding
    todo!(\"Generated encoder for {}\")
}}

", pattern.name, pattern.name
        ));
    }
    
    fs::write(Path::new(out_dir).join("generated_encoders.rs"), code)
        .expect("FATAL: Failed to write generated encoders");
}
```

### 3. Procedural Macro System - REQUIRED IMPLEMENTATION

**YOU MUST CREATE** a new crate `tpde-macros` with EXACTLY this structure:

```toml
# tpde-macros/Cargo.toml
[package]
name = "tpde-macros"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
proc-macro2 = "1.0"
quote = "1.0"
syn = { version = "2.0", features = ["full", "extra-traits"] }
```

```rust
// tpde-macros/src/lib.rs
// THIS IS MANDATORY - IMPLEMENTS THE TEMPLATE FUNCTION ATTRIBUTE

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitStr, Meta};

#[proc_macro_attribute]
pub fn template_function(attrs: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);
    
    // MANDATORY: Extract function metadata
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    
    // MANDATORY: Parse attributes to get target architecture
    let target = extract_target_from_attrs(attrs);
    
    // MANDATORY: Validate target is one of x86_64-v1, v2, v3, or v4
    assert!(matches!(target.as_str(), "x86_64-v1" | "x86_64-v2" | "x86_64-v3" | "x86_64-v4"),
            "Invalid target: {}. Must be one of x86_64-v1, x86_64-v2, x86_64-v3, x86_64-v4", target);
    
    // MANDATORY: Generate both the function AND metadata
    let expanded = quote! {
        // Original function for template extraction
        #[cfg(feature = "template-extraction")]
        #[inline(never)]
        #[no_mangle]
        pub #input_fn
        
        // Metadata for build script
        #[cfg(feature = "template-extraction")]
        #[link_section = ".tpde_template_metadata"]
        #[used]
        static #fn_name: &str = concat!(
            "TPDE_TEMPLATE:",
            #fn_name_str,
            ":",
            #target,
            ";"
        );
    };
    
    TokenStream::from(expanded)
}

// MANDATORY: Helper to extract target architecture
fn extract_target_from_attrs(attrs: TokenStream) -> String {
    // YOU MUST parse attrs and extract target = "..." value
    // MUST support x86_64-v1, x86_64-v2, x86_64-v3, x86_64-v4
    // Default to "x86_64-v1" if not specified
    "x86_64-v1".to_string()
}

// MANDATORY: Macro for generating encoder dispatch
#[proc_macro]
pub fn dispatch_encoder(input: TokenStream) -> TokenStream {
    let opcode = parse_macro_input!(input as LitStr);
    
    // Generate match arm for instruction dispatch
    let expanded = quote! {
        match #opcode {
            "FAdd" => encode_add_f32(encoder, ctx, inputs, output),
            "FSub" => encode_sub_f32(encoder, ctx, inputs, output),
            // ... other opcodes generated from patterns
            _ => Err(EncodingError::UnsupportedInstruction),
        }
    };
    
    TokenStream::from(expanded)
}
```

### 4. Pattern Matching and Optimization - MANDATORY IMPLEMENTATION

**YOU MUST READ** these files to understand pattern extraction:
- `tpde-encodegen/src/x64/Target.cpp` lines 23-977 - Shows EXACT pattern matching for x86-64
- `tpde-encodegen/src/encode_gen.cpp` - Pattern extraction engine
- `src/x64/encoder.rs` lines 1335-1582 - Current manual patterns that MUST be replaced

**YOU MUST CREATE** the pattern matching system EXACTLY as specified:

```rust
// src/codegen/patterns.rs
// THIS FILE IS MANDATORY - DEFINES PATTERN RECOGNITION ENGINE

#[derive(Debug, Clone)]
pub enum Pattern {
    Binary { op: BinaryOp, lhs: Operand, rhs: Operand },
    Unary { op: UnaryOp, operand: Operand },
    Memory { op: MemoryOp, base: Operand, index: Option<Operand>, scale: u8, offset: i32 },
    Select { cond: Operand, true_val: Operand, false_val: Operand },
}

// MANDATORY: Study Target.cpp lines 397-457 for pattern matching logic
// YOU MUST implement handle_rm, handle_rmi, handle_ri patterns
pub const fn matches_pattern(inst: &Instruction, pattern: &Pattern) -> bool {
    match (inst, pattern) {
        (Instruction::Add { .. }, Pattern::Binary { op: BinaryOp::Add, .. }) => true,
        // YOU MUST add ALL patterns from Target.cpp
        _ => false,
    }
}

// MANDATORY: Generate pattern tables from Target.cpp lines 459-733
const PATTERN_TABLE: &[(&str, Pattern)] = &[
    // YOU MUST extract ALL patterns from handle_rm calls in Target.cpp
    ("add_i32", Pattern::Binary { op: BinaryOp::Add, lhs: Operand::Reg, rhs: Operand::RegOrImm }),
    ("load_add_store", Pattern::Memory { op: MemoryOp::LoadModifyStore, .. }),
    // ADD ALL 300+ PATTERNS FROM TARGET.CPP
];

// MANDATORY: Implement pattern conditions from Target.cpp
pub struct PatternCondition {
    pub operand_idx: usize,
    pub condition_type: &'static str, // "encodeable_as_imm32_sext", "encodeable_with"
    pub condition_arg: String,
}

// MANDATORY: Port MICandidate from Target.cpp line 23
pub struct InstructionCandidate {
    pub conditions: Vec<PatternCondition>,
    pub encoder_fn: fn(&mut X64Encoder, &[ValueRef], &mut ScratchReg) -> Result<(), Error>,
}
```

### 5. Generic Encoder Framework - CRITICAL IMPLEMENTATION

**MANDATORY FILES TO READ:**
- `src/x64/encoder.rs` lines 1-1582 - Current encoder that MUST be replaced
- `src/core/value_ref.rs` lines 1-200 - ValueRef types used in encoders
- `src/core/register_file.rs` lines 1-150 - RegisterFile and AsmReg definitions
- `tpde-encodegen/src/x64/EncCompilerTemplate.hpp` - Template for generated encoders

**YOU MUST CREATE** this EXACT encoder framework:

```rust
// src/codegen/encoder.rs
// THIS IS MANDATORY - REPLACES 1500+ LINES OF MANUAL ENCODING

pub trait TemplateEncoder<T> {
    fn encode(&mut self, inputs: &[ValueRef], output: &mut ScratchReg) -> Result<(), Error>;
}

// MANDATORY: Include generated code from build.rs
include!(concat!(env!("OUT_DIR"), "/generated_encoders.rs"));

// MANDATORY: This is the EXACT pattern that build.rs MUST generate
// Study encode_gen.cpp to understand how this is created from templates
impl TemplateEncoder<AddF32> for X64Encoder {
    fn encode(&mut self, inputs: &[ValueRef], output: &mut ScratchReg) -> Result<(), Error> {
        let [lhs, rhs] = inputs else { return Err(Error::InvalidInputs) };
        
        // MANDATORY Pattern 1: Register-Register (from Target.cpp handle_rm)
        if let (Some(lhs_reg), Some(rhs_reg)) = (lhs.try_as_reg(), rhs.try_as_reg()) {
            if output.can_reuse(lhs_reg) {
                self.emit_addss_rr(lhs_reg, rhs_reg)?;
                output.set_reg(lhs_reg);
            } else {
                self.emit_movss_rr(output.alloc_xmm(), lhs_reg)?;
                self.emit_addss_rr(output.reg(), rhs_reg)?;
            }
            return Ok(());
        }
        
        // MANDATORY Pattern 2: Register-Memory fusion (from Target.cpp handle_rm)
        if let Some(mem) = try_fuse_memory_operand(lhs, rhs) {
            self.emit_addss_rm(output.alloc_xmm(), mem)?;
            return Ok(());
        }
        
        // MANDATORY: Fallback path
        self.encode_generic_binary_fp(lhs, rhs, output, Self::emit_addss_rr)
    }
}

// YOU MUST generate encoders for ALL operations in encode_template.c
// This includes 100+ operations that replace 1500+ lines of manual code
```

### 6. Macro-Based DSL for Common Patterns - REQUIRED IMPLEMENTATION

**MANDATORY FILES TO STUDY:**
- `src/llvm/compiler.rs` lines 2800-3300 - Repetitive patterns that MUST be eliminated
- `src/x64/encoder.rs` lines 1335-1582 - Manual encodings to be replaced by macros
- `tpde-encodegen/src/x64/Target.cpp` lines 411-457 - Pattern generation logic

**YOU MUST CREATE** these EXACT macros:

```rust
// src/codegen/macros.rs
// MANDATORY - ELIMINATES 500+ LINES OF REPETITIVE CODE

// MANDATORY: Study Target.cpp handle_rmi for this pattern
macro_rules! define_binary_op {
    ($name:ident, $op:ident, $inst_32:ident, $inst_64:ident) => {
        pub fn $name(&mut self, lhs: ValueRef, rhs: ValueRef, dst: &mut ScratchReg) -> Result<(), Error> {
            binary_op_template!(self, lhs, rhs, dst, {
                32 => self.$inst_32,
                64 => self.$inst_64,
            })
        }
    };
}

// MANDATORY: This template MUST handle all cases from encoder.rs
macro_rules! binary_op_template {
    ($self:expr, $lhs:expr, $rhs:expr, $dst:expr, { $($bits:literal => $inst:expr),+ }) => {{
        let bit_width = $lhs.bit_width();
        match bit_width {
            $($bits => {
                // MANDATORY: Reuse optimization from current implementation
                if $dst.can_reuse($lhs) {
                    $inst($dst.reg(), $rhs.load_to_reg())?;
                } else if $dst.can_reuse($rhs) && is_commutative($inst) {
                    $inst($dst.reg(), $lhs.load_to_reg())?;
                } else {
                    let dst_reg = $dst.alloc(RegBank::for_type($lhs.ty()));
                    $self.mov(dst_reg, $lhs.load_to_reg())?;
                    $inst(dst_reg, $rhs.load_to_reg())?;
                }
                Ok(())
            })+
            _ => Err(Error::UnsupportedBitWidth(bit_width))
        }
    }};
}

// MANDATORY: Define ALL operations using these macros
// This replaces 100+ manual functions in compiler.rs
define_binary_op!(compile_fadd, FAdd, emit_addss_rr, emit_addsd_rr);
define_binary_op!(compile_fsub, FSub, emit_subss_rr, emit_subsd_rr);
define_binary_op!(compile_fmul, FMul, emit_mulss_rr, emit_mulsd_rr);
define_binary_op!(compile_fdiv, FDiv, emit_divss_rr, emit_divsd_rr);

// MANDATORY: Memory operation macros from Target.cpp handle_rm
macro_rules! define_memory_op {
    ($name:ident, $load_inst:ident, $store_inst:ident) => {
        pub fn $name(&mut self, addr: ValueRef, value: Option<ValueRef>) -> Result<ValueRef, Error> {
            // YOU MUST implement pattern from Target.cpp lines 223-366
            todo!("Generated memory operation")
        }
    };
}

// YOU MUST ADD: Shift operations, comparison operations, etc.
// See Target.cpp lines 553-578 for shift patterns
// See Target.cpp lines 535-551 for comparison patterns
```

### 7. Const-Evaluated Instruction Tables - ZERO-COST REQUIREMENT

**MANDATORY READING:**
- `tpde-encodegen/src/x64/Target.cpp` lines 459-733 - ALL x86-64 instructions
- `src/x64/encoder.rs` lines 20-1582 - Current instruction implementations
- Study how C++ version achieves zero runtime overhead

**YOU MUST CREATE** compile-time instruction tables:

```rust
// src/codegen/tables.rs
// MANDATORY - ZERO RUNTIME COST INSTRUCTION LOOKUP

#[derive(Debug, Clone, Copy)]
pub struct InstructionInfo {
    pub opcode: &'static str,
    pub operand_types: &'static [OperandType],
    pub constraints: &'static [Constraint],
    pub size_in_bytes: u8,
}

// MANDATORY: Extract ALL instructions from Target.cpp lines 459-733
// This table MUST be const-evaluated at compile time
pub const INSTRUCTION_TABLE: &[InstructionInfo] = &[
    // YOU MUST add ALL SSE instructions from Target.cpp lines 605-666
    InstructionInfo {
        opcode: "addss",
        operand_types: &[OperandType::XmmReg, OperandType::XmmRegOrMem],
        constraints: &[Constraint::SSE],
        size_in_bytes: 4,
    },
    InstructionInfo {
        opcode: "addsd",
        operand_types: &[OperandType::XmmReg, OperandType::XmmRegOrMem],
        constraints: &[Constraint::SSE2],
        size_in_bytes: 8,
    },
    // YOU MUST ADD ALL 300+ INSTRUCTIONS FROM TARGET.CPP
    // Integer operations: lines 468-551
    // Shift operations: lines 553-578
    // Multiplication/Division: lines 580-595
    // SSE operations: lines 605-666
    // Vector operations: lines 669-730
];

// MANDATORY: Const function for compile-time lookup
// This MUST be evaluated at compile time for zero runtime cost
pub const fn find_instruction(opcode: &str) -> Option<&'static InstructionInfo> {
    let mut i = 0;
    while i < INSTRUCTION_TABLE.len() {
        if const_str_eq(INSTRUCTION_TABLE[i].opcode, opcode) {
            return Some(&INSTRUCTION_TABLE[i]);
        }
        i += 1;
    }
    None
}

// MANDATORY: Const string comparison for compile-time evaluation
const fn const_str_eq(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.len() != b_bytes.len() {
        return false;
    }
    let mut i = 0;
    while i < a_bytes.len() {
        if a_bytes[i] != b_bytes[i] {
            return false;
        }
        i += 1;
    }
    true
}

// MANDATORY: Pattern matching tables from Target.cpp
const PATTERN_VARIANTS: &[(&str, &[&str])] = &[
    // Extract from Target.cpp lines 908-960 for CMOV/SETCC variants
    ("CMOV32", &["CMOVO32rr", "CMOVNO32rr", "CMOVC32rr", /* ... */]),
    ("CMOV64", &["CMOVO64rr", "CMOVNO64rr", "CMOVC64rr", /* ... */]),
    // YOU MUST ADD ALL PATTERN VARIANTS
];
```

### 8. Integration with Existing Compiler - FINAL INTEGRATION

**MANDATORY FILES TO MODIFY:**
- `src/llvm/compiler.rs` lines 2000-3300 - Replace ALL manual implementations
- `src/llvm/compiler.rs` lines 450-650 - Update compile_instruction dispatch
- Read current implementation to preserve correctness guarantees

**YOU MUST MODIFY** the compiler EXACTLY as shown:

```rust
// src/llvm/compiler.rs - MANDATORY CHANGES
impl<'ctx, 'arena> LlvmCompiler<'ctx, 'arena> {
    fn compile_instruction(&mut self, inst: InstructionValue<'ctx>) -> Result<(), Error> {
        // MANDATORY: Replace 3000+ lines with template dispatch
        match inst.get_opcode() {
            InstructionOpcode::FAdd => {
                let inputs = self.prepare_inputs(&inst)?;
                let mut output = ScratchReg::new(&mut self.register_file);
                
                // MANDATORY: Use generated encoder instead of manual implementation
                AddF32::encode(&mut self.encoder, &inputs, &mut output)?;
                
                self.set_result(&inst, output)?;
            }
            InstructionOpcode::FSub => {
                let inputs = self.prepare_inputs(&inst)?;
                let mut output = ScratchReg::new(&mut self.register_file);
                SubF32::encode(&mut self.encoder, &inputs, &mut output)?;
                self.set_result(&inst, output)?;
            }
            // YOU MUST: Replace ALL 100+ opcode handlers with template calls
            // DELETE: compile_fadd_instruction, compile_fsub_instruction, etc.
            // DELETE: 2000+ lines of manual encoding logic
        }
    }
    
    // MANDATORY: Delete ALL manual instruction compilation methods:
    // DELETE: compile_fadd_instruction (lines 2800-2900)
    // DELETE: compile_fsub_instruction (lines 2900-3000)
    // DELETE: compile_fmul_instruction (lines 3000-3100)
    // DELETE: compile_fdiv_instruction (lines 3100-3200)
    // DELETE: ALL other manual methods
    
    // MANDATORY: Add helper for input preparation
    fn prepare_inputs(&mut self, inst: &InstructionValue<'ctx>) -> Result<Vec<ValueRef>, Error> {
        // Extract operands and convert to ValueRefs
        let operands = self.get_instruction_operands(inst)?;
        operands.into_iter()
            .map(|op| self.get_value_ref(op))
            .collect()
    }
}

// MANDATORY: This replaces 3000+ lines with ~100 lines of dispatch code
// The generated encoders handle ALL the complexity that was manual before
```

## Implementation Phases - MANDATORY EXECUTION PLAN

### Phase 1: Foundation (IMMEDIATE - Days 1-3)
**YOU MUST COMPLETE THESE TASKS IN ORDER:**
1. **CREATE** `tpde-macros/` crate with EXACT Cargo.toml from section 3
2. **CREATE** `src/templates/mod.rs` with ALL operations from encode_template.c
3. **IMPLEMENT** `template_function` procedural macro
4. **VERIFY** macro generates correct metadata sections
5. **TEST** with add_f32 template extraction to LLVM IR

### Phase 2: Build System (Days 4-7)
**MANDATORY TASKS:**
1. **REPLACE** existing build.rs with template extraction version
2. **IMPLEMENT** compile_templates_to_ir() for ALL microarchitecture levels
3. **USE** inkwell to parse generated LLVM IR
4. **EXTRACT** instruction patterns using LLVM's codegen
5. **GENERATE** first encoder function for add_f32
6. **VALIDATE** generated code matches manual implementation

### Phase 3: Pattern Extraction (Days 8-14)
**YOU MUST:**
1. **PORT** ALL patterns from Target.cpp lines 23-977
2. **IMPLEMENT** handle_rm, handle_rmi, handle_ri pattern generators
3. **CREATE** pattern matching tables for 300+ instructions
4. **ADD** memory operand fusion detection
5. **IMPLEMENT** immediate constant folding
6. **GENERATE** encoders for ALL SSE/AVX instructions

### Phase 4: Macro System (Days 15-17)
**REQUIRED IMPLEMENTATIONS:**
1. **CREATE** define_binary_op macro for arithmetic operations
2. **CREATE** define_memory_op macro for load/store patterns
3. **CREATE** define_shift_op macro for shift operations
4. **IMPLEMENT** dispatch_encoder procedural macro
5. **VERIFY** macros eliminate 500+ lines of repetitive code

### Phase 5: Full Integration (Days 18-21)
**CRITICAL TASKS:**
1. **DELETE** 3000+ lines from src/llvm/compiler.rs
2. **DELETE** 1500+ lines from src/x64/encoder.rs
3. **REPLACE** with ~100 lines of template dispatch
4. **UPDATE** compile_instruction to use generated encoders
5. **RUN** full test suite to verify correctness
6. **BENCHMARK** compilation speed (MUST be 10-20x faster than LLVM -O0)

## Expected Results - MANDATORY OUTCOMES

### Code Reduction Metrics (MUST ACHIEVE)
- **From 4,500+ lines** → **To ~300 lines** in instruction encoding
- **From 3,000+ lines** → **To ~100 lines** in compiler dispatch
- **Total reduction**: >90% of manual encoding code

### Performance Requirements (MANDATORY)
- **Compilation speed**: 10-20x faster than LLVM -O0
- **Zero runtime overhead**: All pattern matching at compile time
- **Microarchitecture optimization**: Automatic use of best instructions for CPU

### Quality Improvements (MUST DELIVER)
- **Bug reduction**: Template extraction prevents encoding errors
- **Maintainability**: Adding new instruction = writing one template function
- **Correctness**: LLVM validates all generated patterns
- **Coverage**: Support for ALL x86_64 instructions in encode_template.c

### Verification Checklist (ALL MUST PASS)
- [ ] All existing tests pass with generated encoders
- [ ] Generated code matches or exceeds manual performance
- [ ] Support for x86_64-v1 through v4 with appropriate fallbacks
- [ ] Template changes automatically regenerate encoders
- [ ] No manual instruction encoding remains in codebase

## Example: Before and After

### Before (Current Rust - 100+ lines)
```rust
fn compile_fadd_instruction(&mut self, instruction: InstructionValue<'ctx>) -> Result<(), Error> {
    // ... 30 lines of setup ...
    match bit_width {
        32 => {
            if result_reg == left_reg {
                encoder.addss_reg_reg(result_reg, right_reg)?;
            } else if result_reg == right_reg {
                encoder.addss_reg_reg(result_reg, left_reg)?;
            } else {
                encoder.movss_reg_reg(result_reg, left_reg)?;
                encoder.addss_reg_reg(result_reg, right_reg)?;
            }
        }
        64 => {
            // ... similar code for f64 ...
        }
    }
    // ... more code ...
}
```

### After (New Rust - 10 lines)
```rust
// Template definition
#[template_function]
fn add_f32(a: f32, b: f32) -> f32 { a + b }

// Compiler usage
fn compile_fadd_instruction(&mut self, inst: InstructionValue<'ctx>) -> Result<(), Error> {
    let inputs = self.prepare_inputs(&inst)?;
    let mut output = ScratchReg::new(&mut self.register_file);
    AddF32::encode(&mut self.encoder, &inputs, &mut output)?;
    self.set_result(&inst, output)
}
```

## Microarchitecture Support Requirements

**MANDATORY**: The implementation MUST support ALL x86_64 microarchitecture levels:

### x86_64-v1 (Baseline)
- **Features**: SSE, SSE2, x87, CMPXCHG8B, CMOV, FXSR
- **Templates**: Basic floating point, integer operations
- **Target CPU**: x86-64

### x86_64-v2 
- **Features**: SSSE3, SSE4.1, SSE4.2, POPCNT, LAHF/SAHF, CMPXCHG16B
- **Templates**: Population count, advanced SSE operations
- **Target CPU**: x86-64-v2

### x86_64-v3
- **Features**: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE
- **Templates**: FMA operations, leading zero count, 256-bit vectors
- **Target CPU**: x86-64-v3

### x86_64-v4
- **Features**: AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL
- **Templates**: 512-bit vector operations, masked operations
- **Target CPU**: x86-64-v4

**IMPLEMENTATION REQUIREMENTS**:
1. **build.rs MUST** compile each template with the appropriate target-cpu flag
2. **Pattern extraction MUST** handle feature detection for each level
3. **Generated encoders MUST** include runtime CPU feature detection
4. **Fallback paths MUST** exist for unsupported features

```rust
// Example of runtime feature detection in generated code
pub fn encode_fma_f32(a: ValueRef, b: ValueRef, c: ValueRef, out: &mut ScratchReg) -> Result<(), Error> {
    if is_x86_feature_detected!("fma") {
        // Use FMA instruction (x86_64-v3)
        self.emit_vfmadd213ss(out.alloc_xmm(), a.to_xmm(), b.to_xmm(), c.to_xmm())?;
    } else {
        // Fallback to mul+add (x86_64-v1)
        let tmp = self.scratch_xmm();
        self.emit_mulss(tmp, a.to_xmm(), b.to_xmm())?;
        self.emit_addss(out.alloc_xmm(), tmp, c.to_xmm())?;
    }
    Ok(())
}
```

## Conclusion

This approach combines the elegance of the C++ template-based system with Rust's powerful compile-time features, resulting in a solution that is:
- **Idiomatic** - uses Rust's strengths (macros, const eval, type system)
- **Efficient** - zero runtime overhead through compile-time generation
- **Maintainable** - dramatically less code to maintain
- **Extensible** - easy to add new instructions and patterns
- **Safe** - Rust's type system prevents many categories of bugs
- **Portable** - supports all x86_64 microarchitecture levels with appropriate fallbacks

The key insight is that we can achieve the same elegant simplicity as the C++ version while leveraging Rust's unique features to create an even more powerful and flexible system that adapts to different CPU capabilities.

## CRITICAL WARNINGS AND REQUIREMENTS

### DO NOT PROCEED WITHOUT UNDERSTANDING
1. **This is NOT optional** - The current 4,500+ lines of manual encoding is UNMAINTAINABLE
2. **This is NOT a refactor** - This is a COMPLETE REIMPLEMENTATION using template extraction
3. **You MUST follow the C++ pattern** - Study encode_template.c and encodegen extensively
4. **You MUST support ALL microarchitectures** - x86_64-v1 through v4 with runtime detection

### FORBIDDEN ACTIONS
- **DO NOT** attempt to manually port encoders - USE TEMPLATE EXTRACTION
- **DO NOT** skip any instruction from encode_template.c - ALL MUST BE IMPLEMENTED  
- **DO NOT** use generic traits where concrete types work - AVOID OVER-ABSTRACTION
- **DO NOT** ignore CPU feature detection - MUST HAVE FALLBACKS

### MANDATORY VALIDATION
Before considering this complete, you MUST:
1. Run the full test suite with generated encoders
2. Benchmark showing 10-20x speedup over LLVM -O0
3. Demonstrate <500 total lines replacing >4,500 lines
4. Show correct instruction selection for each microarchitecture level
5. Prove zero runtime overhead with assembly inspection

### FINAL REQUIREMENT
**IF YOU CANNOT REDUCE THE CODE BY 90% WHILE MAINTAINING CORRECTNESS AND PERFORMANCE, YOU HAVE FAILED TO UNDERSTAND THE ASSIGNMENT**