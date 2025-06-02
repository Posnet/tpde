//! Encoding snippet generator entry points.
//!
//! `tpde-encodegen` transforms short high level functions (typically written in
//! C and compiled to LLVM IR) into Rust routines that emit the matching machine
//! instructions. The generated code reuses registers when possible and can fuse
//! constant operands or address expressions. By mixing these snippet encoders
//! into a compiler one can implement most instruction selection logic without
//! writing every pattern by hand. See [`tpde_core::overview`] for an
//! extended overview.

use inkwell::{
    context::Context,
    memory_buffer::MemoryBuffer,
    module::Module,
};

/// Parse a text LLVM IR module.
pub fn parse_module<'ctx>(context: &'ctx Context, ir: &str) -> Result<Module<'ctx>, String> {
    let buffer = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), "ir");
    context
        .create_module_from_ir(buffer)
        .map_err(|e| e.to_string())
}

/// Generate Rust source snippets for functions starting with `pattern_`.
pub fn generate_tokens(module: &Module) -> Vec<String> {
    module
        .get_functions()
        .filter_map(|f| {
            let name = f.get_name().to_str().ok()?;
            if let Some(rest) = name.strip_prefix("pattern_") {
                Some(format!("pub fn {}() {{ /* machine code */ }}", rest))
            } else {
                None
            }
        })
        .collect()
}

/// Convenience helper parsing IR text and returning token strings.
pub fn parse_and_generate<'ctx>(
    context: &'ctx Context,
    ir: &str,
) -> Result<Vec<String>, String> {
    let module = parse_module(context, ir)?;
    Ok(generate_tokens(&module))
}

/// Parse the provided LLVM IR module and emit snippet encoders.
#[allow(dead_code)]
pub fn generate(_module: &Module) {
    todo!("encode generation not yet implemented")
}
