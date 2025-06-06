// This README documents the Rust implementation of TPDE, providing an overview of the current
// status (~70-75% complete), architectural design decisions, completed features, and remaining
// work. The document serves as the primary reference for developers working on or evaluating
// the Rust port of TPDE.

# TPDE Rust Implementation

This directory contains the Rust implementation of TPDE (Turbo-Powered Direct Execution), a fast compiler backend framework designed for low-latency compilation with reasonable code quality.

## Status: ~70-75% Complete

The Rust implementation has successfully completed a major architectural redesign that eliminates trait bound complexity and provides direct LLVM integration. It demonstrates superior type safety, memory management, and API design compared to the C++ implementation.

## Architecture

The implementation uses a concrete, LLVM-specific design:

```rust
pub struct LlvmCompiler<'ctx, 'arena> {
    module: &'ctx Module<'ctx>,
    session: &'arena CompilationSession<'arena>,
    value_mgr: ValueAssignment,
    register_file: RegisterFile,
    codegen: FunctionCodegen,
}
```

Key architectural features:
- **Arena-based memory management** using `bumpalo` for efficient allocation
- **Direct LLVM integration** without abstraction overhead
- **Clean opcode dispatch** using `match inst.get_opcode()`
- **Professional testing** with native Rust `#[test]` functions

## Completed Features

### Core Infrastructure ✅
- Value assignment system with multi-part values and reference counting
- Register allocation with clock-based eviction
- ELF object file generation using the `object` crate
- x86-64 instruction encoding with `iced-x86`
- System V x86-64 calling convention

### Instruction Support ✅
- **Arithmetic**: Add, Sub, Mul with proper flag handling
- **Comparison**: ICmp with CMP+SETcc generation
- **Memory**: Load/Store with addressing modes, GEP with LEA optimization
- **Control Flow**: Basic unconditional branches and returns
- **Function Calls**: Direct calls with ABI compliance
- **Stack**: Alloca instruction support

## Missing Features

### Critical Gaps
1. **PHI Node Resolution** - Prevents compilation of loops and complex control flow
2. **Conditional Branches** - No if/else or loop support
3. **Switch Statements** - No multi-way branching
4. **Advanced Calling Convention** - No byval, sret, or varargs support
5. **Exception Handling** - No invoke/landingpad support
6. **Intrinsics** - No LLVM intrinsic functions
7. **ARM64 Backend** - Only x86-64 is implemented

## Building and Testing

```bash
# Build all crates
cargo build --workspace

# Run all tests
cargo test --workspace

# Run specific examples
cargo run --example test_real_codegen
cargo run --example test_function_calls
```

## Project Structure

- `src/core/` - Core compiler infrastructure (value management, register allocation)
- `src/llvm/` - LLVM-specific compiler implementation
- `src/x64/` - x86-64 backend with instruction encoding
- `src/experimental/` - Experimental features and future backends
- `examples/` - Example programs demonstrating compiler usage
- `tests/` - Integration tests

## Development Guidelines

1. **Use arena allocation** - Prefer `bumpalo` allocations tied to compilation sessions
2. **Leverage Rust's type system** - Use `Result` types and proper error handling
3. **Test-driven development** - Write tests alongside new features
4. **Follow C++ patterns** - Reference the C++ implementation for algorithmic guidance
5. **Prioritize correctness** - Focus on generating correct code before optimizing

## Next Steps

The immediate priorities for development are:

1. **Implement PHI node resolution** - Port the C++ algorithm with cycle detection
2. **Add conditional branches** - Enable if/else and loop compilation
3. **Complete memory addressing** - Full base+index*scale+displacement support
4. **Expand test coverage** - More comprehensive LLVM IR test cases

See the main CLAUDE.md file for detailed implementation status and roadmap.