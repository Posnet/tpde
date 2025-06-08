// src/codegen/mod.rs
// Module for template-based code generation

pub mod patterns;
pub mod encoder;

// Include the generated encoders from build.rs
include!(concat!(env!("OUT_DIR"), "/generated_encoders.rs"));