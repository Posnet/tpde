//! Core TPDE framework in Rust.

pub mod adaptor;
pub mod analyzer;
pub mod assembler;
pub mod compiler;

/// Temporary hello world to prove the crate builds.
pub fn hello() -> &'static str {
    "Hello from tpde"
}
