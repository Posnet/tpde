// This module serves as a namespace for experimental and work-in-progress features of the
// TPDE compiler that are not yet production-ready. Currently contains the arm64 module with
// initial AArch64 architecture support. The experimental designation indicates these components
// may have incomplete implementations, limited testing, unstable APIs subject to breaking changes,
// or missing critical functionality. Future experimental features planned include WASM backend
// for WebAssembly target support, RISC-V architecture backend, advanced optimization passes
// beyond the current -O0 level, and alternative register allocation algorithms. Code in this
// module should not be used in production contexts and may be refactored or removed as features
// mature and graduate to the main codebase. The experimental namespace allows for exploration
// of new compilation targets and techniques without affecting the stability of core TPDE functionality.

//! Experimental features and work in progress.
//!
//! This module contains experimental implementations that are not yet
//! ready for production use. Features may be incomplete, untested, or
//! subject to significant API changes.

pub mod arm64;

// Future experimental features:
// - WASM backend
// - RISC-V support
// - Advanced optimizations
// - Alternative register allocators
