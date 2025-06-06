# Logging in TPDE Rust Implementation

## Overview

The TPDE Rust implementation uses the `log` crate for internal debug messages and `env_logger` for log output control. This provides better control over debug output compared to using `println!` everywhere.

## Usage

To see log output when running examples or tests, set the `RUST_LOG` environment variable:

```bash
# Show all log messages
RUST_LOG=debug cargo run --example test_real_codegen

# Show only info level and above
RUST_LOG=info cargo run --example gep_test

# Show trace level (most verbose)
RUST_LOG=trace cargo test

# Filter by module
RUST_LOG=tpde::llvm_compiler=debug cargo run --example test_function_calls

# Multiple filters
RUST_LOG=tpde::llvm_compiler=debug,tpde::x64_encoder=trace cargo test
```

## Log Levels

The following log levels are used throughout the codebase:

- **trace**: Very detailed information (instruction bytes, register states)
- **debug**: Debugging information (compilation steps, value tracking)
- **info**: General information (function compilation, major milestones)
- **warn**: Warnings (unimplemented features, fallbacks)
- **error**: Errors (compilation failures)

## Examples

### Running with debug output:
```bash
RUST_LOG=debug cargo run --example test_real_codegen
```

### Running tests with trace output:
```bash
RUST_LOG=trace cargo test test_factorial_compilation
```

### Filtering specific modules:
```bash
RUST_LOG=tpde::llvm_compiler_concrete=trace,tpde::register_file=debug cargo test
```

## Implementation Notes

- Internal debug messages use `log::debug!`, `log::trace!`, etc.
- User-facing output (CLI tools, test results) still uses `println!`
- All example programs now initialize `env_logger` at startup
- Test output uses `println!` for test results, `log::*` for internal state

## Common Patterns

```rust
// In main() or at start of examples
env_logger::init();

// Debug output
log::debug!("Compiling function: {}", func_name);

// Trace output for detailed state
log::trace!("Register allocation: {:?} -> {:?}", value, reg);

// Info for major steps
log::info!("✅ Successfully compiled {} functions", count);

// Warnings for incomplete features
log::warn!("Intrinsic '{}' not yet implemented", name);
```