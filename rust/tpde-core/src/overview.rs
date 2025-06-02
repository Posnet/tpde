//! Overview of the TPDE framework.
//!
//! This summary condenses the design docs located under `docs/tpde`. The TPDE
//! framework is a modular compiler back-end focusing on low latency code
//! generation. It is structured around seven key pieces:
//!
//! 1. **IRAdaptor** – user supplied bridge exposing an SSA IR to the compiler.
//! 2. **Analyzer** – computes block order and value liveness information.
//! 3. **CompilerBase** – drives register allocation and code generation.
//! 4. **Arch Compiler** – architecture specific helpers for prologue/epilogue,
//!    spills and instruction emission.
//! 5. **Assembler** – owns sections and symbols and produces ELF objects or
//!    executable memory.
//! 6. **Instruction Selection** – user provided code implementing the mapping
//!    from IR instructions to machine code.
//! 7. **Snippet Encoders** – optional helpers generated from high level code to
//!    reduce manual instruction selection.
//!
//! ```text
//! ┌───────────┐      ┌───────────────────────────┐     ┌───────────┐
//! │           │◄─────┼                           ├────►│           │
//! │ IRAdaptor │      │     CompilerBase          │     │ Assembler │
//! │           │      │                           │     │           │
//! └───────────┘      │              ┌────────┐   │     └───────────┘
//!                    │              │Analyzer│   │                  
//!                    │              └────────┘   │                  
//!                    └─┬───▲──────────┬──▲────▲──┘                  
//!                      │   │          │  │    │                     
//!                      │   │          │  │    │                     
//!                  ┌───▼───┴──────┐   │  │  ┌─┴─────────────────┐   
//!                  │ Compiler for │◄──┼──┼──┤ Encoding Snippets │   
//!                  │    <arch>    │   │  │  │     (optional)    │   
//!                  └──────┬─▲─────┘   │  │  └─▲─────────────────┘   
//!                         │ │         │  │    │                     
//!                         │ │         │  │    │                     
//!                         │ │         │  │    │                     
//!                       ┌─▼─┴─────────▼──┴────┴─────┐               
//!                       │                           │               
//!                       │ Compiler specific to <IR> │               
//!                       │                           │               
//!                       └───────────────────────────┘                  
//! ```
//! 
//! The standard compilation flow is:
//!
//! - The adaptor enumerates functions.
//! - For each function the analyzer derives block layout in reverse post order
//!   and tracks liveness for every value.
//! - Architecture specific code uses this information while emitting machine
//!   instructions and managing register allocation.
//! - The assembler collects sections, relocations and unwind data and finally
//!   writes an ELF object or maps the code into memory.
//!
//! TPDE is intended for baseline code generation. It trades heavy
//! optimization passes for very fast compile times while still delivering
//! reasonable code quality comparable to `-O0` builds. Vector operations and
//! less common IR constructs may not be well supported. When the input IR is
//! outside of TPDE's comfort zone one is expected to fall back to an
//! optimizing back-end such as LLVM or Cranelift.
//!
//! Porting to another architecture generally involves implementing the
//! architecture compiler and assembler and optionally providing snippet
//! encoders for common patterns.
