//! Compiler reference overview.
//!
//! `CompilerBase` orchestrates the transformation from the user's IR into machine code.
//! The original documentation describes a template heavy class hierarchy. In Rust we
//! model this with generic structs and traits.
//!
//! High level duties:
//! - Maintain the register allocator state across blocks.
//! - Manage stack slots and unwind information.
//! - Delegate instruction emission to architecture specific code or user provided selection.
//!
//! The compiler interacts with three helpers:
//! 1. The `IrAdaptor` trait for iterating functions and instructions.
//! 2. `Analyzer` for liveness and block order.
//! 3. An architecture `Assembler` owning sections and symbols.
//!
//! Typical flow:
//! ```text
//! for each function {
//!     adaptor.switch_func(func);
//!     analyzer.switch_func(&mut adaptor, func);
//!     architecture.prologue();
//!     for each block in analyzer.order() {
//!         for inst in adaptor.block_insts(block) {
//!             user_compiler.compile_inst(inst);
//!         }
//!     }
//!     architecture.epilogue();
//! }
//! ```
//!
//! Values are tracked via `ValueRef` handles. When the reference count drops
//! to zero the allocator may reuse the registers or spill to the stack if necessary.
//! Additional helpers like `ScratchReg` will be ported from the C++ code later.

use crate::{adaptor::IrAdaptor, analyzer::Analyzer, assembler::Assembler};

/// Architecture independent compiler driver.
///
/// [`CompilerBase`] coordinates the entire compilation pipeline described in the
/// TPDE documentation.  In broad strokes it asks the
/// [`IrAdaptor`] for functions, runs [`Analyzer`] to compute liveness and block
/// layout, then hands off to architecture specific code to emit instructions via
/// an [`Assembler`].  Register allocation happens on the fly based on the
/// liveness info.  This file only implements a thin skeleton so far.
#[allow(dead_code)]
pub struct CompilerBase<A: IrAdaptor, ASM: Assembler<A>> {
    adaptor: A,
    analyzer: Analyzer<A>,
    assembler: ASM,
}

impl<A: IrAdaptor, ASM: Assembler<A>> CompilerBase<A, ASM> {
    /// Create a new compiler base from an adaptor and assembler.
    pub fn new(adaptor: A, assembler: ASM) -> Self {
        Self {
            adaptor,
            analyzer: Analyzer::new(),
            assembler,
        }
    }

    /// Compile all functions provided by the adaptor.
    pub fn compile(&mut self) -> bool {
        let funcs: Vec<_> = self.adaptor.funcs().collect();
        for func in funcs {
            if !self.adaptor.switch_func(func) {
                continue;
            }
            self.analyzer.switch_func(&mut self.adaptor, func);
            // architecture specific code generation would go here
        }
        true
    }
}
