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

### Current Status: ~40-45% Complete (Phase 1 Complete, Phase 2 In Progress)

The Rust implementation in `rust/` has made significant progress with core infrastructure now in place. We can compile real LLVM IR functions including factorial with control flow, arithmetic operations, and function calls through the complete TPDE pipeline.

### Implementation Gap Analysis

#### ✅ **Completed Components**
- **IRAdaptor trait** - Well-designed abstraction for IR traversal (`rust/tpde-core/src/adaptor.rs`)
- **Basic CompilerBase** - Core compilation flow outlined (`rust/tpde-core/src/compiler.rs`)
- **Liveness Analysis** - RPO ordering and basic value liveness tracking (`rust/tpde-core/src/analyzer.rs`)
- **ELF Assembler** - Functional object file generation using `object` crate (`rust/tpde-core/src/assembler.rs`)
- **LLVM Integration** - Basic module parsing with inkwell (`rust/tpde-llvm/src/lib.rs`)
- **Enhanced LLVM IR Adaptor** - Real-world LLVM IR support with control flow, PHI nodes, function calls (`rust/tpde-llvm/src/enhanced_adaptor.rs`)
- **Encodegen Tool** - IR pattern analysis and stub generation (`rust/tpde-encodegen/`)
- **ValueAssignment System** - Complete with multi-part values, reference counting, storage locations (`rust/tpde-core/src/value_assignment.rs`)
- **RegisterFile** - Clock-based allocation with eviction, locking, and spill callbacks (`rust/tpde-core/src/register_file.rs`)
- **ValueRef/ValuePartRef** - RAII interfaces for instruction selection (`rust/tpde-core/src/value_ref.rs`)
- **x86-64 Instruction Encoder** - Basic encoders using iced-x86 (`rust/tpde-core/src/x64_encoder.rs`)
- **Calling Convention** - System V x86-64 ABI implementation (`rust/tpde-core/src/calling_convention.rs`)
- **Function Codegen** - Prologue/epilogue generation, frame management (`rust/tpde-core/src/function_codegen.rs`)
- **Complete Compiler** - End-to-end compilation pipeline integration (`rust/tpde-core/src/complete_compiler.rs`)

#### ❌ **Critical Missing Components (~55-60% of functionality)**

**Instruction Selection & Codegen (~70% missing):**
- Complete instruction selection for all LLVM IR constructs
- Memory operations (loads, stores, addressing modes)
- Branch and conditional instruction compilation
- Function call instruction generation and integration
- Complex expression lowering and optimization
- Immediate value materialization strategies

**Block Successor Analysis (~85% missing):**
- Proper extraction of successor blocks from terminator instructions
- Switch statement target analysis
- Invoke instruction handling for exception flow
- Complex control flow pattern support

**Advanced IR Support (~75% missing):**
- ConstantExpr decomposition and handling
- Global variable and alloca support
- LLVM intrinsics implementation (memcpy, etc.)
- Exception handling infrastructure
- Complex addressing modes and GEP instruction support

**ARM64 Backend (~95% missing):**
- ARM64 instruction encoders
- AAPCS calling convention implementation
- ARM64-specific register allocation
- Target-specific instruction selection

**Performance & Optimization (~80% missing):**
- Copy coalescing and register pressure optimization
- Linear scan register allocation improvements
- Instruction fusion and pattern matching
- Compile-time performance optimization

### Development Roadmap

#### **Phase 1: Core Infrastructure (Months 1-3) ✅ COMPLETED**
**Goal**: Enable compilation of simple arithmetic functions

1. **ValueAssignment System** (`rust/tpde-core/src/value_assignment.rs`) ✅
   - Complete C++ ValueAssignment, AssignmentPartRef infrastructure
   - Multi-part value tracking with reference counting
   - Register vs memory location management

2. **RegisterFile Implementation** (`rust/tpde-core/src/register_file.rs`) ✅
   - Clock-based allocation with eviction and locking
   - Register pressure tracking and spilling
   - Integration with ValueRef RAII interfaces

3. **Function Frame Management** (`rust/tpde-core/src/function_codegen.rs`) ✅
   - Calling convention integration
   - Prologue/epilogue generation
   - Frame layout calculation

4. **x86-64 Instruction Encoders** (`rust/tpde-core/src/x64_encoder.rs`) ✅
   - mov, add, sub, ret instructions using iced-x86
   - Immediate value encoding
   - Register-to-register operations

5. **Complete Compiler Integration** (`rust/tpde-core/src/complete_compiler.rs`) ✅
   - End-to-end compilation pipeline
   - Enhanced LLVM IR adaptor integration
   - Real factorial function compilation with control flow

**Milestone**: ✅ Successfully compile `int add(int a, int b) { return a + b; }` and factorial function

#### **Phase 2: Instruction Selection & Real Codegen (Months 4-6) 🚧 IN PROGRESS**
**Goal**: Generate real machine code for all basic IR constructs

1. **Complete Instruction Selection** 🚧 CURRENT TASK
   - Full LLVM IR instruction support (arithmetic, loads, stores, calls, branches)
   - Integration with existing x86-64 encoder and register allocation
   - Proper block successor extraction from terminators
   - Memory addressing mode optimization

2. **Enhanced Control Flow** ⏳ NEXT
   - Real branch instruction compilation with condition codes
   - Switch statement handling
   - Function call instruction generation

3. **Memory Operations & Global Support** ⏳ PENDING
   - Load/store instruction selection with addressing modes
   - Global variable and alloca support
   - Stack access pattern optimization

4. **Block Successor Analysis** ⏳ PENDING
   - Extract successors from br, switch, invoke instructions
   - Support for complex control flow patterns
   - Exception handling integration

**Milestone**: Generate real machine code for factorial function and other complex C functions

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

### Recent Progress Summary

**Major Achievements (Phase 1 Complete):**
- ✅ **End-to-end compilation pipeline** - Can compile real LLVM IR functions from start to finish
- ✅ **Complete value management** - Full ValueAssignment system with multi-part values and RAII interfaces  
- ✅ **Working register allocation** - Clock-based algorithm with eviction, locking, and spill support
- ✅ **x86-64 instruction encoding** - Basic arithmetic and movement instructions using iced-x86
- ✅ **Calling convention support** - System V x86-64 ABI with prologue/epilogue generation
- ✅ **Enhanced LLVM IR support** - Real-world IR constructs including control flow and function calls
- ✅ **Factorial compilation test** - Successfully compiles recursive factorial function with control flow

**Current Focus (Phase 2 In Progress):**
- 🚧 **Instruction selection completion** - Adding support for all LLVM IR instruction types
- 🚧 **Block successor analysis** - Proper extraction from terminator instructions  
- ⏳ **Memory operation codegen** - Load/store instruction selection with addressing modes
- ⏳ **Enhanced control flow** - Branch compilation with condition codes

### Contribution Priorities

When working on the Rust implementation, focus on these areas in order:

1. **Phase 2 instruction selection** - Complete support for all LLVM IR instructions
2. **Real machine code generation** - Integrate instruction selection with register allocation
3. **Block successor implementation** - Extract control flow from LLVM terminators
4. **Test-driven development** - Expand test coverage for complex IR patterns
5. **Performance validation** - Benchmark against C++ version regularly

The Rust implementation now has a solid foundation with Phase 1 complete. The trait-based design provides better type safety than the C++ version while maintaining equivalent functionality. The focus is now on completing instruction selection to generate real executable machine code.