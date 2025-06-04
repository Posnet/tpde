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

## Rust Implementation Status

### Current Status: ~15-20% Complete

The Rust implementation in `rust/` provides a promising foundation with trait-based abstractions that mirror the C++ concepts, but is missing most critical functionality needed for actual compilation.

### Implementation Gap Analysis

#### ✅ **Completed Components**
- **IRAdaptor trait** - Well-designed abstraction for IR traversal (`rust/tpde-core/src/adaptor.rs`)
- **Basic CompilerBase** - Core compilation flow outlined (`rust/tpde-core/src/compiler.rs`)
- **Liveness Analysis** - RPO ordering and basic value liveness tracking (`rust/tpde-core/src/analyzer.rs`)
- **ELF Assembler** - Functional object file generation using `object` crate (`rust/tpde-core/src/assembler.rs`)
- **LLVM Integration** - Basic module parsing with inkwell (`rust/tpde-llvm/src/lib.rs`)
- **Encodegen Tool** - IR pattern analysis and stub generation (`rust/tpde-encodegen/`)

#### ❌ **Critical Missing Components (~80-85% of functionality)**

**Register Allocation & Value Management (~90% missing):**
- Complete ValueAssignment system with multi-part value support
- RegisterFile with lock counts, fixed assignments, clock-based allocation
- AssignmentPartRef for efficient register allocation tracking
- ScratchReg infrastructure for temporary register management
- Stack slot allocation with free lists and spill management
- PHI node resolution with cycle detection

**Instruction Selection (~95% missing):**
- Real machine code generation (current: placeholder stubs)
- x86-64/ARM64 instruction encoders
- Address calculation and memory operation encoding
- Immediate value materialization and optimization
- Branch and call instruction handling
- Complex expression lowering

**Calling Conventions (~100% missing):**
- Function prologue/epilogue generation
- Argument passing (registers vs stack placement)
- Return value handling according to ABI
- System V x86-64 and ARM64 AAPCS support
- Unwind information generation (.eh_frame)

**Advanced IR Support (~85% missing):**
- ConstantExpr decomposition and handling
- Global variable and alloca support
- LLVM intrinsics implementation
- Complex control flow (switches, indirect calls)
- Exception handling infrastructure

### Development Roadmap

#### **Phase 1: Core Infrastructure (Months 1-3)**
**Goal**: Enable compilation of simple arithmetic functions

1. **ValueAssignment System** (`rust/tpde-core/src/value_assignment.rs`)
   - Port C++ ValueAssignment, AssignmentPartRef infrastructure
   - Implement multi-part value tracking
   - Add register vs memory location management

2. **RegisterFile Implementation** (`rust/tpde-core/src/register_file.rs`)
   - Basic register allocation with lock counts
   - Clock-based allocation algorithm
   - Register pressure tracking and spilling

3. **Stack Frame Management** (`rust/tpde-core/src/stack_frame.rs`)
   - Stack slot allocation
   - Spill slot management
   - Frame layout calculation

4. **Basic x86-64 Instruction Encoders** (`rust/tpde-llvm/src/x64/encoders.rs`)
   - mov, add, sub, ret instructions
   - Immediate value encoding
   - Register-to-register operations

**Milestone**: Compile `int add(int a, int b) { return a + b; }`

#### **Phase 2: Basic Function Compilation (Months 4-6)**
**Goal**: Handle function calls and control flow

1. **Calling Convention Implementation**
   - System V x86-64 ABI support
   - Function prologue/epilogue generation
   - Argument register allocation
   - Return value handling

2. **Memory Operations**
   - Load/store instruction selection
   - Address calculation optimization
   - Stack access patterns

3. **Control Flow**
   - Branch instruction encoding
   - Basic block transitions
   - Simple conditional compilation

4. **Enhanced LLVM Backend**
   - Better constant handling
   - Support for allocas and globals
   - Basic intrinsics (memcpy, etc.)

**Milestone**: Compile functions with local variables, calls, and branches

#### **Phase 3: Advanced Features (Months 7-9)**
**Goal**: Handle complex IR constructs

1. **PHI Node Resolution**
   - Cycle detection and breaking
   - Register allocation across merges
   - Optimization of PHI copies

2. **Complex Instruction Selection**
   - Pattern matching for compound operations
   - Addressing mode optimization
   - Instruction fusion opportunities

3. **JIT Execution Support**
   - Memory mapping for generated code
   - Runtime symbol resolution
   - Unwind information registration

4. **ARM64 Backend**
   - Port x86-64 work to ARM64
   - AAPCS calling convention
   - ARM64-specific optimizations

**Milestone**: Feature parity with basic C++ functionality

#### **Phase 4: Optimization & Completeness (Months 10-12)**
**Goal**: Production-ready implementation

1. **Advanced Register Allocation**
   - Linear scan improvements
   - Copy coalescing
   - Register pressure heuristics

2. **Performance Optimization**
   - Compile-time performance tuning
   - Memory allocation optimization
   - Benchmark against C++ version

3. **Comprehensive Testing**
   - LLVM IR compatibility test suite
   - Performance regression tests
   - Fuzzing and stress testing

4. **Documentation & Tooling**
   - API documentation
   - Developer guides
   - Integration examples

### Development Guidelines

#### **Architecture Principles**
- **Maintain trait-based design** - Rust's type system provides better safety than C++ concepts
- **Preserve performance characteristics** - Must match C++ compile-time performance
- **Use safe Rust where possible** - Only use `unsafe` for performance-critical register allocation
- **Comprehensive testing** - Each component should have unit and integration tests

#### **Implementation Notes**
- **Value tracking**: The C++ `ValLocalIdx` system needs careful Rust translation
- **Register allocation**: May require `unsafe` blocks for performance-critical paths
- **Memory management**: Use `Box`/`Vec` instead of C++ custom allocators initially
- **Error handling**: Leverage Rust's `Result` types for better error propagation
- **LLVM integration**: inkwell provides safer interface than direct C++ API

#### **Critical Success Factors**
1. **Register allocation correctness** - Core to TPDE's value proposition
2. **Compile-time performance** - Must maintain 10-20x speedup over LLVM -O0
3. **ABI compliance** - Generated code must work with standard calling conventions
4. **Test coverage** - Comprehensive validation against C++ implementation

### Contribution Priorities

When working on the Rust implementation, focus on these areas in order:

1. **Phase 1 components** - Core infrastructure needed for any compilation
2. **Test-driven development** - Write failing tests first, then implement
3. **Performance validation** - Benchmark against C++ version regularly
4. **Documentation** - Keep architecture decisions well-documented
5. **Incremental validation** - Ensure each component works before moving on

The Rust implementation has excellent architectural foundations but requires significant development effort to reach production readiness. The trait-based design provides better type safety than the C++ version, making the investment worthwhile for long-term maintainability.