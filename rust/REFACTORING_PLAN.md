# TPDE Rust Module Refactoring Plan

## Overview

This plan addresses the current naming confusion and architectural holdovers in the TPDE Rust codebase. The goal is to make the primary LLVM compilation path obvious while properly organizing legacy and experimental code.

## Current Issues

### 1. Confusing Module Names
- `basic.rs` vs `enhanced.rs` - Suggests quality levels when they're different approaches
- `CompleteCompiler` - Generic trait-based (mostly unused)
- `llvm_compiler_concrete.rs` - The actual implementation but poorly named
- Multiple analyzer versions with unclear purposes

### 2. Architectural Debt
- Phase 2B redesign eliminated trait bounds but naming hasn't caught up
- "Enhanced" vs "Basic" distinction no longer meaningful
- "Concrete" suffix suggests abstraction problems

### 3. Module Organization
- Primary compilation path not obvious
- Legacy code mixed with production code
- 4000+ line files that need splitting

## Proposed Structure

### Phase 1: Core Reorganization

```
src/
├── llvm/                       # Primary LLVM compilation path (was llvm_compiler_concrete.rs)
│   ├── mod.rs                  # Main LlvmCompiler implementation
│   ├── adaptor.rs              # LLVM IR adaptor (was enhanced.rs)
│   ├── instruction_select.rs   # Instruction selection logic
│   ├── phi_resolution.rs       # PHI node handling (moved from phi_resolver.rs)
│   ├── function_analysis.rs    # Function analysis (best of current analyzers)
│   └── tests.rs                # LLVM-specific tests
│
├── core/                       # Shared components
│   ├── mod.rs
│   ├── register_file.rs        # Register allocation
│   ├── value_assignment.rs     # Value tracking
│   ├── session.rs              # Compilation session (was compilation_session.rs)
│   ├── error.rs                # Error types
│   └── assembler.rs            # ELF generation
│
├── x64/                        # Architecture-specific
│   ├── mod.rs
│   ├── encoder.rs              # x86-64 instruction encoding (was x64_encoder.rs)
│   ├── backend.rs              # x86-64 backend (was x64_backend.rs)
│   └── calling_convention.rs   # System V ABI
│
├── legacy/                     # Deprecated but kept for reference
│   ├── mod.rs
│   ├── generic_compiler.rs     # (was complete_compiler.rs)
│   ├── minimal_adaptor.rs      # (was basic.rs)
│   └── trait_based.rs          # Old trait-based approach
│
├── experimental/               # Work in progress
│   ├── mod.rs
│   └── arm64/                  # Future ARM64 support
│
└── lib.rs                      # Public API
```

### Phase 2: Naming Changes

| Current Name | New Name | Rationale |
|-------------|----------|-----------|
| `llvm_compiler_concrete.rs` | `llvm/mod.rs` | It's THE LLVM compiler, not a "concrete" variant |
| `enhanced.rs` | `llvm/adaptor.rs` | It's the production adaptor, not "enhanced" |
| `basic.rs` | `legacy/minimal_adaptor.rs` | Clarify it's not for production use |
| `CompleteCompiler` | `legacy::GenericCompiler` | It's the old generic approach |
| `compilation_session.rs` | `core/session.rs` | Shorter, clearer |
| `phi_resolver.rs` | `llvm/phi_resolution.rs` | Part of LLVM compilation |
| `function_analyzer_arena.rs` | `llvm/function_analysis.rs` | The arena version is the good one |

### Phase 3: Type Consolidation

Move shared types to appropriate modules:
- `AddressingMode` → `x64/addressing.rs`
- `GepExpression` → `llvm/gep.rs`
- `InstructionCategory` → `llvm/instruction.rs`

## Migration Strategy

### Step 1: Create New Structure (Week 1)
```rust
// src/llvm/mod.rs
pub use crate::llvm_compiler_concrete::*;  // Temporary re-export

// src/legacy/mod.rs
#[deprecated(note = "Use llvm::LlvmCompiler instead")]
pub use crate::complete_compiler::CompleteCompiler as GenericCompiler;
```

### Step 2: Update Imports (Week 2)
```rust
// Old
use tpde::llvm_compiler_concrete::LlvmCompiler;
use tpde::complete_compiler::CompleteCompiler;

// New
use tpde::llvm::LlvmCompiler;
use tpde::legacy::GenericCompiler;  // If still needed
```

### Step 3: Move Files (Week 3)
- Git mv operations to preserve history
- Update internal imports
- Fix tests

### Step 4: Cleanup (Week 4)
- Remove re-exports
- Delete unused code
- Update documentation

## Benefits

1. **Clarity**: `llvm::LlvmCompiler` is obviously the main implementation
2. **Organization**: Related code grouped together
3. **Discoverability**: New developers immediately see the primary path
4. **Maintenance**: Legacy code clearly marked
5. **Future-proof**: Room for ARM64 and other architectures

## Backwards Compatibility

### Deprecated Aliases
```rust
// lib.rs - temporary compatibility layer
#[deprecated(since = "0.2.0", note = "Use llvm::LlvmCompiler")]
pub use llvm::LlvmCompiler as LlvmCompilerConcrete;

#[deprecated(since = "0.2.0", note = "Use legacy::GenericCompiler")]
pub use legacy::GenericCompiler as CompleteCompiler;
```

### Documentation Update
```rust
//! # TPDE - Fast SSA-based IR Compilation
//! 
//! Primary usage:
//! ```
//! use tpde::llvm::{LlvmCompiler, LlvmAdaptor};
//! use tpde::core::CompilationSession;
//! ```
```

## Testing Strategy

1. **Before refactoring**: Ensure all tests pass
2. **During refactoring**: Run tests after each file move
3. **After refactoring**: Add integration tests for new module structure
4. **Deprecation tests**: Ensure old imports still work with warnings

## Timeline

- Week 1: Create new directory structure, add compatibility layer
- Week 2: Update all imports in examples and tests  
- Week 3: Move files to new locations
- Week 4: Remove compatibility layer, update docs

## Success Criteria

1. Primary LLVM path obvious to new developers
2. No more "enhanced" vs "basic" confusion
3. Legacy code clearly separated
4. All tests passing
5. Examples use new imports
6. Documentation updated

## Future Considerations

This structure supports:
- Adding ARM64 support in `experimental/arm64/`
- Moving experimental to production (e.g., `experimental/arm64/` → `arm64/`)
- Adding more architectures without cluttering the main path
- Potential `no_std` support by isolating `std` dependencies