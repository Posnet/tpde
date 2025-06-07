// This module represents the experimental ARM64/AArch64 backend for TPDE, providing initial
// infrastructure for compiling to 64-bit ARM processors. Currently this is a placeholder
// module with no implementation, serving as the foundation for future AArch64 support.
// The planned implementation will include AAPCS64 (ARM Architecture Procedure Call Standard)
// calling convention with X0-X7 for arguments, X0/X1 for returns, and X19-X28 as callee-saved;
// ARMv8-A instruction encoding for generating native ARM64 machine code; register allocation
// adapted for 31 general-purpose registers (X0-X30) and 32 SIMD registers (V0-V31); proper
// handling of the link register (X30/LR) and frame pointer (X29/FP); and integration with
// the existing TPDE infrastructure including IrAdaptor, Backend trait, and ELF assembler.
// This experimental status indicates the ARM64 backend is not yet functional and will require
// significant development to reach feature parity with the x86-64 backend. The module structure
// mirrors x64/ with planned submodules for encoder, calling_convention, backend, and function_codegen.

//! ARM64/AArch64 backend (experimental).
//!
//! This module will contain the ARM64 implementation of TPDE.
//!
//! Target features:
//! - AAPCS64 calling convention
//! - ARMv8-A instruction encoding
//! - Integration with existing TPDE infrastructure
