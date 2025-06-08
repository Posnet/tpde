TPDE C++ Backend Design and Implementation Guide
================================================

**TPDE** is a fast, low-latency compiler backend framework written in C++20. It supports multiple SSA-based source IRs (e.g., LLVM) and targets ELF object emission for x86-64 and AArch64. TPDE’s design emphasizes template-driven, IR-agnostic infrastructure, high performance, and pragmatic extensibility. This document describes the TPDE C++ architecture and implementation principles in sufficient detail for a full reimplementation.

---

Table of Contents
-----------------
1. High-Level Architecture
2. Core Components and Dataflow
3. IRAdaptor Pattern
4. Compiler Core
   - Analyzer Pass
   - Register Allocation & Value Assignment
   - Calling Convention Assignment
5. Target Backends
   - Assembler (ELF, DWARF, Relocation)
   - x64 and a64 Backend Structure
6. Memory and Value Management
7. Testing and Extensibility
8. Appendix: Build and Integration

---

## 1. High-Level Architecture

- **IR-agnostic core**: The majority of logic is independent of the specific incoming IR.
- **Template/final static polymorphism**: The core orchestrates compilation using templates with CRTP and C++20 concepts.
- **Directory structure**
  - `tpde/include/tpde/` – Core headers; target-specific subdirs for x64/arm64.
  - `tpde/src/` – Implementations and extension points.
  - Each component is logically isolated and built for reuse/extensibility.

---

## 2. Core Components and Dataflow

**Major Steps**:
1. IR input adapter implements IRAdaptor concept.
2. Analyzer pass computes block order, loops, liveness.
3. CompilerBase orchestrates register assignment, stack management, spill/fill, and overall codegen.
4. Target Backend emits ELF file with architecture-appropriate assembler and ABI logic.

**Key modules/files:**
- `IRAdaptor.hpp` – IR concept interface
- `Analyzer.hpp` – Block ordering, loop/tree analysis
- `CompilerBase.hpp` – Main orchestrator (template)
- `RegisterFile.hpp`, `ValueAssignment.hpp`, `AssignmentPartRef.hpp` – Register state, value allocation, part tracking
- `AssemblerElf.hpp`, `arm64/AssemblerElfA64.hpp`, `x64/AssemblerElfX64.hpp` – ELF and platform assembler backends

---

## 3. IRAdaptor Pattern

- **Goal**: Decouple TPDE internals from any specific IR, allowing integration with LLVM, custom kernels, or research IRs.
- **Approach**: Define a *concept*; implementors provide:
  - Type aliases for Value, Inst, Block, Func (by lightweight ref, typically ≤8 bytes).
  - Iterators/ranges for blocks, values, and PHI/meta info.
  - Methods to access operands, PHI nodes, block predecessors, etc.

**Excerpt** (see `IRAdaptor.hpp`):
```cpp
template <typename T>
concept IRAdaptor = requires(T a) {
  typename T::IRValueRef;
  typename T::IRInstRef;
  typename T::IRBlockRef;
  typename T::IRFuncRef;
  // block/value iterators, PHI access, meta slots, ...
};
```

- For each new source IR, implement a single IRAdaptor and plug into CompilerBase.

---

## 4. Compiler Core

### Analyzer Pass

- Computes:
  - Block layout (linear order for code emission)
  - Loop structure (for block layout/pack)
  - Liveness and lifetime of values

- Central structure (from `Analyzer.hpp`):
```cpp
struct Analyzer<Adaptor> {
  Adaptor *adaptor;
  SmallVector<IRBlockRef, N> block_layout;
  SmallVector<Loop, 16> loops;
  ...
};
```

### Register Allocation & Value Assignment

- **ValueAssignment**
  - Describes *where* (registers/stack) a value lives at a program point
  - Handles splitting for SIMD or aggregates
  - Includes fast state transitions (register <-> stack, dirty/clean)
  - Allocated using a freelist/bump strategy for performance

- **RegisterFile**
  - Bitmask model for available registers, by bank (GPR, SIMD, ...)
  - Exposes inline bit ops for allocation, freeing, and tracking

- **AssignmentPartRef**
  - Lightweight pointer/slicer into ValueAssignment, tracks location/state/flags on a per-part basis

### Calling Convention Assignment

- **CCAssigner** (abstract base; backends specialize)
  - Assigns registers/stack for arguments and return
  - Handles varargs, byval, sret, alignment, bank assignment
  - Implements per-platform ABI logic via tables or inline code
  - Exposed through the backend’s Compiler config

---

## 5. Target Backends
### Automatic Instruction Encoding Generation: Encodegen

TPDE employs an innovative "encodegen" step to automate the production of instruction encoding helpers for different architectures:

#### Step-by-Step Example (AArch64 `add x0, x1, x2`)
1. **Reference C Template Function**
   - Write a C function that describes the operation:
   ```c
   // templates_arm64.c
   unsigned add_rr(unsigned x, unsigned y) { return x + y; }
   ```
2. **Compile to Object File**
   - Use the target's C compiler (or cross-compiler):
   ```sh
   aarch64-linux-gnu-gcc -O2 -c templates_arm64.c -o templates_arm64.o
   ```
3. **Run the `tpde-encodegen` Tool**
   - This tool parses the object file, locates functions, and extracts the bytes encoding each instruction.
   - For this function, it extracts the encoding for `add x0, x1, x2`.
4. **Generate C++ Encoders**
   - The tool writes:
   ```cpp
   constexpr uint32_t encode_add_rr(unsigned dst, unsigned n, unsigned m) {
     return 0x8b000000 | (m << 16) | (n << 5) | dst;
   }
   ```
5. **Assembler Inclusion and Use**
   - These helpers are included in platform assembler headers
   - Backend macros or encode routines use these helpers for emission

**Why do this?**
- Guarantees encodings match what compilers emit
- Easily extensible for new instructions or platforms—write new reference C or tweak templates only
- Eliminates risk of hand-coded bitfield errors


### Assembler (ELF, DWARF, relocation)

- **AssemblerElfBase**
  - Target-neutral ELF emission: section headers, program headers, .text/.data, DWARF FDE/CIE, relocations
  - Section writers use a bump allocator, allow for fast append, alignment, fixup
  - Instantiated (CRTP) for x64/a64, each providing ELF/DWARF constants & fixup logic

### x64 and a64 Backend Structure

- **x64**
  - `AssemblerElfX64` and `CompilerX64` provide instruction encoding and ABI for x86-64
  - Define per-arch CCAssigner, register sets, unwind table params, fixup types
  - Macros for fast instruction encoding (see header defines)

- **a64**
  - Analogous to x64; adapts to AArch64 instruction encoding and ABI rules
  - Handles extra cases (long branch veneers, ARM64 unwinding particulars)

---

## 6. Memory and Value Management

- Uses custom **BumpAllocator** for most short-lived allocations
- ValueAssignments allocated via freelist/arena for speed
- **SmallVector** and **SmallBitSet** minimize heap trips by providing in-place growth

---

## 7. Testing and Extensibility

- **TestIR**: Minimal built-in IR for testing codegen/analysis, used in core test runner
- Pluggable: Add new IRs by providing an IRAdaptor; add new targets by cloning/adjusting x64/a64 compilers and assembler backends
- New calling conventions plug in via CCAssigner

---

## 8. Appendix: Build and Integration

- Requires C++20, CMake3.23+, standard platform headers; fadec-enc2.h (x64) and disarm64.h (a64) for encoding
- To build:
```sh
cmake -B build
cmake --build build
```
- To run core tests:
```sh
cmake --build build --target check-tpde
```

---

### Reference: Major APIs

| Module/Class           | Purpose                              |
|------------------------|--------------------------------------|
| IRAdaptor              | IR plug-in interface                  |
| Analyzer               | Block layout and liveness analysis    |
| CompilerBase           | Orchestrates codegen and assignments  |
| ValueAssignment        | Stores value-to-location assignments  |
| AssignmentPartRef      | Fast part-specific assignment access  |
| RegisterFile           | Per-bank register bookkeeping         |
| AssemblerElfBase       | ELF/DWARF emission                    |
| AssemblerElfX64/A64    | x64/AArch64 backends                  |
| CompilerX64/A64        | Platform-specific generation          |

---

This guide covers the core design and implementation patterns inside the TPDE C++ backend. Reference code, function signatures, and further annotations can be found throughout the `tpde/include/tpde/` and `tpde/src/` directories.