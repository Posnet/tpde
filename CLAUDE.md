# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TPDE is a fast compiler back-end framework that adapts to existing SSA IRs, designed for low-latency compilation with reasonable code quality (targeting `-O0` level). It compiles 10-20x faster than LLVM -O0 with similar code quality and targets ELF-based x86-64 and AArch64 platforms.

## Build System

### Core Build Commands

```bash
# Configure and build (from project root)
cmake -B build
cmake --build build

# Build with specific options
cmake -B build -DTPDE_ENABLE_LLVM=ON -DTPDE_ENABLE_ENCODEGEN=ON -DTPDE_INCLUDE_TESTS=ON
cmake --build build

# Build documentation (optional)
cmake -B build -DTPDE_BUILD_DOCS=ON
cmake --build build
```

### Testing

```bash
# Run all tests
cmake --build build --target check-tpde

# Run specific test suites
cmake --build build --target check-tpde-core    # Core TPDE tests
cmake --build build --target check-tpde-llvm    # LLVM backend tests

# Legacy test targets (backwards compatibility)
cmake --build build --target tpde_filetest      # Core tests
cmake --build build --target tpde_llvm_filetest # LLVM tests
```

### Individual Tools

```bash
# Build and run specific tools
./build/tpde/tpde_test                          # Core test executable
./build/tpde-llvm/tpde-llc <input.ll>          # LLVM-to-native compiler
./build/tpde-llvm/tpde-lli <input.ll>          # LLVM interpreter/JIT
./build/tpde-encodegen/tpde_encodegen <input>  # Code generator utility
```

### Rust Components

```bash
# From rust/ directory
cargo build --workspace
cargo test --workspace

# Individual crates
cd rust/tpde-core && cargo build
cd rust/tpde-encodegen && cargo build  
cd rust/tpde-llvm && cargo build
```

## Architecture Overview

TPDE is structured around three core concepts connected through well-defined interfaces:

### 1. IRAdaptor Pattern
- **Purpose**: Provides a generic interface for different SSA IRs to plug into TPDE
- **Key Files**: `tpde/include/tpde/IRAdaptor.hpp`
- **Concept**: Templates define requirements for IR integration (values, blocks, functions, iteration)
- **Implementation**: Each IR (like LLVM) implements the IRAdaptor concept

### 2. Compiler Framework
- **Purpose**: Core compilation logic independent of source IR and target architecture  
- **Key Files**: `tpde/include/tpde/Compiler.hpp`, `tpde/include/tpde/CompilerBase.hpp`
- **Architecture**: Template-based design using C++20 concepts for type safety
- **Components**:
  - Value assignment and register allocation
  - Code generation orchestration
  - Platform-independent optimization passes

### 3. Target Architecture Backends
- **x86-64**: `tpde/include/tpde/x64/` and `tpde-llvm/src/x64/`
- **AArch64**: `tpde/include/tpde/arm64/` and `tpde-llvm/src/arm64/`
- **ELF Assembly**: `tpde/include/tpde/AssemblerElf.hpp` with target-specific implementations

### Component Structure

- **tpde/**: Core framework (IR-agnostic compiler, ELF generation, target backends)
- **tpde-llvm/**: LLVM IR adaptor and tools (`tpde-llc`, `tpde-lli`)
- **tpde-encodegen/**: Utility to generate target code from LLVM Machine IR
- **rust/**: Experimental Rust implementations of core components

### Key Design Patterns

- **C++20 Concepts**: Extensive use for compile-time interface validation
- **Template Specialization**: Platform-specific code generation without runtime overhead
- **Dependency Injection**: IRAdaptor and Compiler are composed rather than inherited
- **RAII**: Resource management through RAII patterns, no exceptions used

## Development Notes

### Dependencies
- LLVM 19.1+ (for tpde-llvm and tpde-encodegen components)
- CMake 3.23+
- C++20 compiler
- Git submodules: fadec, disarm, spdlog, small_vector

### Configuration Options
- `TPDE_ENABLE_LLVM`: Enable LLVM backend (default: ON)
- `TPDE_ENABLE_ENCODEGEN`: Enable encodegen tool (default: ON)  
- `TPDE_X64`: Enable x86-64 support (default: ON)
- `TPDE_A64`: Enable AArch64 support (default: ON)
- `TPDE_INCLUDE_TESTS`: Enable testing (default: ON)
- `TPDE_ENABLE_COVERAGE`: Enable coverage instrumentation (default: OFF)

### Logging and Debugging
- Logging controlled by `TPDE_LOGGING` (DebugOnly/ON/OFF)
- Debug builds enable asserts via `TPDE_ASSERTS`
- Address sanitizer enabled in debug mode for top-level builds