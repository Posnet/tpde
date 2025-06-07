# LLVM Instruction Implementation Status

This document tracks the implementation status of LLVM instructions in the Rust TPDE compiler and provides a roadmap for completing missing functionality.

## Implementation Overview

- **Total LLVM Instructions**: ~60+ core instructions
- **Currently Implemented**: 22 instructions (35%)
- **Critical Missing**: 40+ instructions (65%)

## ✅ Completed Instructions

### Arithmetic Operations
- [x] `Add` - Integer addition (with LEA optimization)
- [x] `Sub` - Integer subtraction  
- [x] `Mul` - Integer multiplication

### Bitwise Operations
- [x] `And` - Bitwise AND
- [x] `Or` - Bitwise OR
- [x] `Xor` - Bitwise XOR
- [x] `Shl` - Shift left
- [x] `LShr` - Logical shift right
- [x] `AShr` - Arithmetic shift right

### Comparison Operations
- [x] `ICmp` - Integer comparison (all predicates)

### Memory Operations
- [x] `Load` - Load from memory
- [x] `Store` - Store to memory
- [x] `Alloca` - Stack allocation
- [x] `GetElementPtr` - Pointer arithmetic (with LEA optimization)

### Control Flow
- [x] `Br` - Branch (conditional and unconditional)
- [x] `Switch` - Multi-way branch
- [x] `Return` - Function return
- [x] `Call` - Direct function calls
- [x] `Phi` - PHI nodes (basic support, simplified cycle detection)

### Type Conversions
- [x] `SExt` - Sign extend
- [x] `ZExt` - Zero extend
- [x] `Trunc` - Truncate

## ❌ Missing Instructions

### Priority 1: Critical for Basic C (Implement First)

#### Integer Division/Remainder
- [ ] `UDiv` - Unsigned division
- [ ] `SDiv` - Signed division  
- [ ] `URem` - Unsigned remainder
- [ ] `SRem` - Signed remainder

**Implementation Notes**:
- Use x86-64 DIV/IDIV instructions
- Handle division by zero
- Manage RDX:RAX register pair for division

#### Select Instruction
- [ ] `Select` - Conditional select (ternary operator)

**Implementation Notes**:
- Can use CMOVcc instructions on x86-64
- Alternative: use CMP + SETcc + AND/OR pattern

#### Basic Floating Point
- [ ] `FAdd` - Floating point addition
- [ ] `FSub` - Floating point subtraction
- [ ] `FMul` - Floating point multiplication
- [ ] `FDiv` - Floating point division
- [ ] `FCmp` - Floating point comparison

**Implementation Notes**:
- Use SSE instructions (ADDSS/ADDSD, etc.)
- Handle XMM register allocation
- Support both float and double

### Priority 2: Extended Arithmetic & Conversions

#### Floating Point Operations
- [ ] `FRem` - Floating point remainder
- [ ] `FNeg` - Floating point negation

#### Integer/Float Conversions
- [ ] `FPToSI` - Float to signed integer
- [ ] `FPToUI` - Float to unsigned integer
- [ ] `SIToFP` - Signed integer to float
- [ ] `UIToFP` - Unsigned integer to float
- [ ] `FPTrunc` - Float truncation (double to float)
- [ ] `FPExt` - Float extension (float to double)

#### Pointer/Integer Conversions
- [ ] `PtrToInt` - Pointer to integer
- [ ] `IntToPtr` - Integer to pointer
- [ ] `BitCast` - Bitwise cast (reinterpret)

### Priority 3: Advanced Control Flow

#### Exception Handling
- [ ] `Invoke` - Call with exception handling
- [ ] `LandingPad` - Exception landing pad
- [ ] `Resume` - Resume exception propagation
- [ ] `CatchPad` - Catch exception
- [ ] `CleanupPad` - Cleanup on exception

#### Other Control Flow
- [ ] `IndirectBr` - Indirect branch (computed goto)
- [ ] `Unreachable` - Mark unreachable code

### Priority 4: Memory & Concurrency

#### Atomic Operations
- [ ] `AtomicRMW` - Atomic read-modify-write
- [ ] `AtomicCmpXchg` - Atomic compare and exchange
- [ ] `Fence` - Memory fence/barrier

#### Memory Attributes
- [ ] Volatile load/store support
- [ ] Atomic load/store support
- [ ] Alignment attributes

### Priority 5: Advanced Data Operations

#### Vector Operations
- [ ] `ExtractElement` - Extract from vector
- [ ] `InsertElement` - Insert into vector
- [ ] `ShuffleVector` - Shuffle vector elements

#### Aggregate Operations
- [ ] `ExtractValue` - Extract from struct/array
- [ ] `InsertValue` - Insert into struct/array

### Priority 6: Specialized Instructions

#### Address Space
- [ ] `AddrSpaceCast` - Cast between address spaces

#### Special Values
- [ ] `Freeze` - Freeze poison/undef values

#### Variable Arguments
- [ ] `VAArg` - Extract vararg parameter

## Implementation Plan

### Phase 1: Essential C Support (1-2 weeks)
1. **Integer Division/Remainder** (2-3 days)
   - Implement UDiv/SDiv with x86-64 DIV/IDIV
   - Add register allocation for RDX:RAX pair
   - Handle division by zero cases
   - Add comprehensive tests

2. **Select Instruction** (1-2 days)
   - Implement using CMOVcc instructions
   - Add fallback for older CPUs
   - Test with various types

3. **Basic Floating Point** (3-4 days)
   - Implement FAdd/FSub/FMul/FDiv
   - Add XMM register support to register allocator
   - Implement FCmp with proper NaN handling
   - Test with float and double types

### Phase 2: Type System Completeness (1 week)
1. **Float/Int Conversions** (2-3 days)
   - Implement all 6 conversion instructions
   - Handle edge cases (overflow, NaN)
   - Use CVTSS2SI, CVTSI2SS, etc.

2. **Pointer Conversions** (1-2 days)
   - Implement PtrToInt/IntToPtr
   - Add BitCast support
   - Ensure proper 64-bit handling

3. **FRem and FNeg** (1 day)
   - FNeg using XOR with sign bit
   - FRem using FPREM instruction or libcall

### Phase 3: Advanced Features (2 weeks)
1. **Aggregate Operations** (2-3 days)
   - Implement ExtractValue/InsertValue
   - Handle nested structs/arrays
   - Optimize for simple cases

2. **Exception Handling** (3-4 days)
   - Basic Invoke/LandingPad support
   - Integrate with platform unwinding
   - Add Resume instruction

3. **Atomic Operations** (3-4 days)
   - Implement AtomicRMW operations
   - Add AtomicCmpXchg
   - Implement Fence with proper barriers

4. **Vector Operations** (2-3 days)
   - Basic ExtractElement/InsertElement
   - ShuffleVector with common patterns
   - Integrate with SSE/AVX

### Phase 4: Optimization & Polish (1 week)
1. **Inline Assembly** (2-3 days)
   - Parse inline asm syntax
   - Register allocation integration
   - Constraint handling

2. **Intrinsics** (3-4 days)
   - Start with essential intrinsics (memcpy, memset)
   - Add math intrinsics (sqrt, sin, cos)
   - Platform-specific intrinsics

3. **Global Variables** (2 days)
   - Global variable access
   - Thread-local storage
   - Constant pools

## Testing Strategy

For each instruction implementation:
1. **Unit Tests**: Test instruction in isolation
2. **Integration Tests**: Test with real LLVM IR from C code
3. **Comparison Tests**: Compare output with LLVM's llc
4. **Edge Cases**: Test boundary conditions, special values
5. **Performance Tests**: Ensure generated code is efficient

## File Organization

```
src/llvm/compiler.rs
├── compile_instruction()          # Main dispatch
├── Integer Arithmetic
│   ├── compile_div_instruction()  # NEW: UDiv/SDiv
│   └── compile_rem_instruction()  # NEW: URem/SRem
├── Floating Point
│   ├── compile_fadd_instruction() # NEW
│   ├── compile_fsub_instruction() # NEW
│   ├── compile_fmul_instruction() # NEW
│   ├── compile_fdiv_instruction() # NEW
│   └── compile_fcmp_instruction() # NEW
├── Conversions
│   ├── compile_fptosi_instruction() # NEW
│   ├── compile_fptoui_instruction() # NEW
│   ├── compile_sitofp_instruction() # NEW
│   ├── compile_uitofp_instruction() # NEW
│   └── compile_bitcast_instruction() # NEW
└── Advanced
    ├── compile_select_instruction()  # NEW
    ├── compile_extractvalue_instruction() # NEW
    └── compile_atomic_instruction()  # NEW
```

## Dependencies

- **x64 Encoder**: Need to add missing instructions
  - DIV/IDIV for division
  - SSE instructions for floating point
  - CMOVcc for select
  - LOCK prefix for atomics

- **Register Allocator**: Need XMM register support
  - Add XMM0-XMM15 to bank 1
  - Handle FP calling convention
  - Manage callee-saved XMM registers

- **Type System**: Need to handle more types
  - Floating point types (f32, f64)
  - Vector types
  - Pointer address spaces

## Success Criteria

The implementation will be considered complete when:
1. Can compile standard C library functions
2. Can compile floating point benchmarks
3. Can compile multi-threaded C programs
4. Passes all C++ implementation tests
5. Performance within 2x of LLVM -O0

## Notes

- Start with x86-64 only, ARM64 can come later
- Prioritize correctness over performance initially  
- Reuse patterns from existing instructions
- Maintain comprehensive test coverage
- Document any deviations from LLVM semantics