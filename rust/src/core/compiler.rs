// This module defines the core compiler framework with CompilerBase and Backend trait.
// CompilerBase orchestrates the entire compilation pipeline: it coordinates between the
// IrAdaptor (for IR access), Analyzer (for liveness/block ordering), and architecture-specific
// Backend implementations. The compilation flow iterates through functions, computes liveness
// info, generates prologue, compiles each instruction in RPO block order, and emits epilogue.
// The Backend trait provides hooks for architecture-specific code generation including
// prologue/epilogue generation and per-instruction compilation. Register allocation happens
// on-the-fly based on liveness information. The design uses unsafe pointers to work around
// Rust's borrow checker limitations when the backend needs mutable access to the compiler
// base during instruction compilation. This is a simplified skeleton of the C++ template-heavy
// architecture adapted to Rust's trait system.

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

use super::{adaptor::IrAdaptor, analyzer::Analyzer, assembler::Assembler, session::CompilationSession};

/// Hooks implemented by architecture specific compiler code.
///
/// The `Backend` drives instruction selection and can emit a prologue
/// and epilogue around each function.  Methods receive a mutable reference
/// to the [`CompilerBase`] so they can use register allocation helpers.
pub trait Backend<A: IrAdaptor, ASM: Assembler<A>> {
    fn gen_prologue<'arena>(&mut self, base: &mut CompilerBase<'arena, A, ASM, Self>)
    where
        Self: Sized;
    fn gen_epilogue<'arena>(&mut self, base: &mut CompilerBase<'arena, A, ASM, Self>)
    where
        Self: Sized;
    fn compile_inst<'arena>(&mut self, base: &mut CompilerBase<'arena, A, ASM, Self>, inst: A::InstRef) -> bool
    where
        Self: Sized;
}

/// Architecture independent compiler driver.
///
/// [`CompilerBase`] coordinates the entire compilation pipeline described in the
/// TPDE documentation.  In broad strokes it asks the
/// [`IrAdaptor`] for functions, runs [`Analyzer`] to compute liveness and block
/// layout, then hands off to architecture specific code to emit instructions via
/// an [`Assembler`].  Register allocation happens on the fly based on the
/// liveness info.  This file only implements a thin skeleton so far.
#[allow(dead_code)]
pub struct CompilerBase<'arena, A: IrAdaptor, ASM: Assembler<A>, C: Backend<A, ASM>> {
    pub adaptor: A,
    pub analyzer: Analyzer<'arena, 'arena, A>,
    pub assembler: ASM,
    backend: C,
    _marker: std::marker::PhantomData<&'arena ()>,
}

impl<'arena, A, ASM, C> CompilerBase<'arena, A, ASM, C>
where
    A: IrAdaptor,
    ASM: Assembler<A>,
    C: Backend<A, ASM>,
{
    /// Create a new compiler base from an adaptor, assembler and backend.
    pub fn new(adaptor: A, assembler: ASM, backend: C, session: &'arena CompilationSession<'arena>) -> Self {
        Self {
            adaptor,
            analyzer: Analyzer::new(session),
            assembler,
            backend,
            _marker: std::marker::PhantomData,
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

            // Using a raw pointer avoids borrow checker conflicts between the
            // backend and the base structure.
            let base_ptr: *mut Self = self;
            let backend = &mut self.backend;
            unsafe { backend.gen_prologue(&mut *base_ptr) };

            for block in self.adaptor.blocks() {
                for inst in self.adaptor.block_insts(block) {
                    let ok = unsafe { backend.compile_inst(&mut *base_ptr, inst) };
                    if !ok {
                        return false;
                    }
                }
            }

            unsafe { backend.gen_epilogue(&mut *base_ptr) };
        }

        self.assembler.finalize();
        true
    }
}
