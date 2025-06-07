# Elegant Code Generation for TPDE Rust

## Overview

This document outlines a plan to transform the Rust TPDE implementation from manual instruction encoding to an elegant template-based approach similar to the C++ version, while leveraging Rust's unique features for an idiomatic solution.

## Current State Analysis

### Problem
The current Rust implementation requires hundreds of lines of manual encoding for each instruction type:
- Manual implementation of every instruction variant (addss, addsd, subss, subsd, etc.)
- Explicit handling of register allocation and moves
- Repetitive patterns across similar operations
- Easy to miss optimization opportunities

### C++ Solution
The C++ version elegantly solves this by:
1. Writing simple template functions in C
2. Compiling them to LLVM IR/Machine code
3. Extracting the generated sequences
4. Auto-generating encoder functions

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

### 1. Template Definition Layer

Create a template definition system using Rust's type system:

```rust
// src/templates/mod.rs
#![allow(dead_code)]

use core::arch::x86_64::*;

#[template_function(target = "x86_64-v1")]
pub fn add_f32(a: f32, b: f32) -> f32 {
    a + b
}

#[template_function(target = "x86_64-v1")]
pub fn add_f64(a: f64, b: f64) -> f64 {
    a + b
}

#[template_function(target = "x86_64-v1", inline_mode = "always")]
pub fn select_i32(cond: bool, a: i32, b: i32) -> i32 {
    if cond { a } else { b }
}

// More complex patterns with addressing modes
#[template_function(target = "x86_64-v1")]
pub unsafe fn load_add_store_i32(ptr: *mut i32, value: i32) -> i32 {
    let old = *ptr;
    *ptr = old + value;
    old
}

// SIMD operations using Rust intrinsics
#[template_function(target = "x86_64-v2")]
pub unsafe fn add_v4f32(a: __m128, b: __m128) -> __m128 {
    _mm_add_ps(a, b)
}
```

### 2. Build-Time Extraction (build.rs)

Implement a sophisticated build.rs that:

```rust
// build.rs
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/templates/");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let target = env::var("TARGET").unwrap();
    
    // Step 1: Compile templates to LLVM IR
    compile_templates_to_ir(&out_dir, &target);
    
    // Step 2: Extract machine code patterns
    let patterns = extract_instruction_patterns(&out_dir);
    
    // Step 3: Generate encoder implementations
    generate_encoders(&patterns, &out_dir);
    
    // Step 4: Generate pattern matching tables
    generate_pattern_tables(&patterns, &out_dir);
}

fn compile_templates_to_ir(out_dir: &str, target: &str) {
    // Use rustc directly to compile with specific flags
    Command::new("rustc")
        .args(&[
            "--crate-type=rlib",
            "--emit=llvm-ir",
            "-C", "opt-level=3",
            "-C", "target-cpu=native",
            "-C", "no-redzone=yes",
            "--target", target,
            "src/templates/mod.rs",
            "-o", &format!("{}/templates.ll", out_dir),
        ])
        .status()
        .expect("Failed to compile templates");
}

struct InstructionPattern {
    name: String,
    signature: FunctionSignature,
    instructions: Vec<MachineInst>,
    constraints: Vec<Constraint>,
}

fn extract_instruction_patterns(out_dir: &str) -> Vec<InstructionPattern> {
    // Parse LLVM IR and extract patterns
    // This is where we'd use llvm-sys or inkwell to analyze the IR
    // and extract the actual instruction sequences
    todo!()
}
```

### 3. Procedural Macro System

Create procedural macros for template analysis and code generation:

```rust
// tpde-macros/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn template_function(attrs: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);
    let attrs = parse_macro_input!(attrs as TemplateAttrs);
    
    // Analyze function signature and body
    let analysis = analyze_template_function(&input_fn, &attrs);
    
    // Generate metadata for build-time extraction
    let metadata = generate_template_metadata(&analysis);
    
    // Keep original function and add metadata
    let expanded = quote! {
        #[cfg(feature = "template-extraction")]
        #input_fn
        
        #metadata
    };
    
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn generate_encoder(input: TokenStream) -> TokenStream {
    let pattern = parse_macro_input!(input as EncoderPattern);
    
    // Generate efficient encoder based on pattern
    let encoder = generate_encoder_impl(&pattern);
    
    TokenStream::from(encoder)
}
```

### 4. Pattern Matching and Optimization

Leverage Rust's const evaluation and pattern matching:

```rust
// src/codegen/patterns.rs
#[derive(Debug, Clone)]
pub enum Pattern {
    Binary { op: BinaryOp, lhs: Operand, rhs: Operand },
    Unary { op: UnaryOp, operand: Operand },
    Memory { op: MemoryOp, base: Operand, index: Option<Operand>, scale: u8, offset: i32 },
    Select { cond: Operand, true_val: Operand, false_val: Operand },
}

// Const functions for pattern matching at compile time
pub const fn matches_pattern(inst: &Instruction, pattern: &Pattern) -> bool {
    match (inst, pattern) {
        (Instruction::Add { .. }, Pattern::Binary { op: BinaryOp::Add, .. }) => true,
        // ... more patterns
        _ => false,
    }
}

// Generate pattern matching tables at compile time
const PATTERN_TABLE: &[(&str, Pattern)] = &[
    ("add_i32", Pattern::Binary { op: BinaryOp::Add, lhs: Operand::Reg, rhs: Operand::RegOrImm }),
    ("load_add_store", Pattern::Memory { op: MemoryOp::LoadModifyStore, .. }),
    // ... more patterns
];
```

### 5. Generic Encoder Framework

Create a generic framework that works with the generated patterns:

```rust
// src/codegen/encoder.rs
pub trait TemplateEncoder<T> {
    fn encode(&mut self, inputs: &[ValueRef], output: &mut ScratchReg) -> Result<(), Error>;
}

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/generated_encoders.rs"));

// Example of generated code:
impl TemplateEncoder<AddF32> for X64Encoder {
    fn encode(&mut self, inputs: &[ValueRef], output: &mut ScratchReg) -> Result<(), Error> {
        let [lhs, rhs] = inputs else { return Err(Error::InvalidInputs) };
        
        // Pattern 1: Both in registers
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
        
        // Pattern 2: Memory operand fusion
        if let Some(mem) = try_fuse_memory_operand(lhs, rhs) {
            self.emit_addss_rm(output.alloc_xmm(), mem)?;
            return Ok(());
        }
        
        // Fallback
        self.encode_generic_binary_fp(lhs, rhs, output, Self::emit_addss_rr)
    }
}
```

### 6. Macro-Based DSL for Common Patterns

Create ergonomic macros for defining common instruction patterns:

```rust
// src/codegen/macros.rs
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

macro_rules! binary_op_template {
    ($self:expr, $lhs:expr, $rhs:expr, $dst:expr, { $($bits:literal => $inst:expr),+ }) => {{
        let bit_width = $lhs.bit_width();
        match bit_width {
            $($bits => {
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

// Usage:
define_binary_op!(compile_fadd, FAdd, emit_addss_rr, emit_addsd_rr);
define_binary_op!(compile_fsub, FSub, emit_subss_rr, emit_subsd_rr);
```

### 7. Const-Evaluated Instruction Tables

Use const evaluation for zero-cost abstractions:

```rust
// src/codegen/tables.rs
#[derive(Debug, Clone, Copy)]
pub struct InstructionInfo {
    pub opcode: &'static str,
    pub operand_types: &'static [OperandType],
    pub constraints: &'static [Constraint],
    pub size_in_bytes: u8,
}

// Build compile-time tables
pub const INSTRUCTION_TABLE: &[InstructionInfo] = &[
    InstructionInfo {
        opcode: "addss",
        operand_types: &[OperandType::XmmReg, OperandType::XmmRegOrMem],
        constraints: &[Constraint::SSE],
        size_in_bytes: 4,
    },
    // ... hundreds more
];

// Const function to look up instructions at compile time
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
```

### 8. Integration with Existing Compiler

Modify the existing compiler to use the new template system:

```rust
// src/llvm/compiler.rs
impl<'ctx, 'arena> LlvmCompiler<'ctx, 'arena> {
    fn compile_instruction(&mut self, inst: InstructionValue<'ctx>) -> Result<(), Error> {
        // Use the template encoder system
        match inst.get_opcode() {
            InstructionOpcode::FAdd => {
                let inputs = self.prepare_inputs(&inst)?;
                let mut output = ScratchReg::new(&mut self.register_file);
                
                // Use generated encoder
                AddF32::encode(&mut self.encoder, &inputs, &mut output)?;
                
                self.set_result(&inst, output)?;
            }
            // ... other opcodes
        }
    }
}
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
1. Create template module structure
2. Implement basic build.rs for template compilation
3. Set up procedural macro crate
4. Create proof-of-concept for one instruction (add_f32)

### Phase 2: Extraction Engine (Week 3-4)
1. Implement LLVM IR parsing in build.rs
2. Create pattern extraction logic
3. Generate first encoder automatically
4. Validate against manual implementation

### Phase 3: Pattern System (Week 5-6)
1. Implement full pattern matching system
2. Create const-evaluated tables
3. Add memory operand fusion
4. Support immediate folding

### Phase 4: Macro DSL (Week 7)
1. Design and implement instruction definition macros
2. Create binary/unary operation templates
3. Add support for complex patterns (select, memory ops)

### Phase 5: Integration (Week 8)
1. Replace manual encoders with generated ones
2. Update compiler to use new system
3. Performance testing and optimization
4. Documentation

## Benefits

### Compared to Current Rust Implementation
- **90% less code** for instruction encoding
- **Automatic optimization** from LLVM's codegen
- **Fewer bugs** from manual encoding
- **Easier to add new instructions**

### Compared to C++ Implementation
- **Type safety** with Rust's type system
- **Better error messages** from procedural macros
- **No external tooling** - all integrated with Cargo
- **Const evaluation** for zero-runtime overhead
- **More flexible** pattern matching system

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

## Conclusion

This approach combines the elegance of the C++ template-based system with Rust's powerful compile-time features, resulting in a solution that is:
- **Idiomatic** - uses Rust's strengths (macros, const eval, type system)
- **Efficient** - zero runtime overhead through compile-time generation
- **Maintainable** - dramatically less code to maintain
- **Extensible** - easy to add new instructions and patterns
- **Safe** - Rust's type system prevents many categories of bugs

The key insight is that we can achieve the same elegant simplicity as the C++ version while leveraging Rust's unique features to create an even more powerful and flexible system.