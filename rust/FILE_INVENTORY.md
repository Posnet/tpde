# TPDE Rust File Inventory

This document categorizes all files in the Rust implementation based on the recent module refactoring.

## Legend
- ✅ **POST-REFACTOR**: Files created or updated as part of the refactoring
- 📦 **PRE-REFACTOR**: Original files that should be moved/updated
- ⚠️ **LEGACY**: Files that should be marked as deprecated or removed
- 🔧 **CORE**: Essential infrastructure files

## New Module Structure (POST-REFACTOR)

### ✅ `src/llvm/mod.rs`
**Status**: POST-REFACTOR  
**Purpose**: Main module file for LLVM compilation. Re-exports LlvmCompiler and related types.  
**Action**: Keep as primary entry point

### ✅ `src/core/mod.rs`
**Status**: POST-REFACTOR  
**Purpose**: Re-exports shared infrastructure (session, registers, values, error types)  
**Action**: Keep for shared components

### ✅ `src/x64/mod.rs`
**Status**: POST-REFACTOR  
**Purpose**: Re-exports x86-64 specific components (encoder, backend, calling convention)  
**Action**: Keep for architecture-specific code

### ✅ `src/legacy/mod.rs`
**Status**: POST-REFACTOR  
**Purpose**: Contains deprecated implementations with compatibility aliases  
**Action**: Keep for migration period

### ✅ `src/experimental/mod.rs` & `src/experimental/arm64/mod.rs`
**Status**: POST-REFACTOR  
**Purpose**: Placeholder for future experimental features and ARM64 support  
**Action**: Keep for future development

## Files to be Moved/Refactored

### 📦 `src/llvm_compiler_concrete.rs`
**Status**: PRE-REFACTOR (should move to `src/llvm/compiler.rs`)  
**Purpose**: Main LLVM compiler implementation with direct opcode dispatch  
**Action**: Move to `src/llvm/compiler.rs`

### 📦 `src/llvm_adaptor/enhanced.rs`
**Status**: PRE-REFACTOR (should move to `src/llvm/adaptor.rs`)  
**Purpose**: Production LLVM IR adaptor with full feature support  
**Action**: Move to `src/llvm/adaptor.rs`, remove "enhanced" terminology

### 📦 `src/phi_resolver.rs`
**Status**: PRE-REFACTOR (should move to `src/llvm/phi_resolution.rs`)  
**Purpose**: PHI node resolution with cycle detection  
**Action**: Move to `src/llvm/phi_resolution.rs`

### 📦 `src/compilation_session.rs`
**Status**: PRE-REFACTOR (should move to `src/core/session.rs`)  
**Purpose**: Arena-based compilation session management  
**Action**: Move to `src/core/session.rs`

### 📦 `src/register_file.rs`
**Status**: PRE-REFACTOR (should move to `src/core/register_file.rs`)  
**Purpose**: Register allocation and tracking  
**Action**: Move to `src/core/register_file.rs`

### 📦 `src/value_assignment.rs`
**Status**: PRE-REFACTOR (should move to `src/core/value_assignment.rs`)  
**Purpose**: Value storage and assignment management  
**Action**: Move to `src/core/value_assignment.rs`

### 📦 `src/error.rs`
**Status**: PRE-REFACTOR (should move to `src/core/error.rs`)  
**Purpose**: Common error types  
**Action**: Move to `src/core/error.rs`

### 📦 `src/assembler.rs`
**Status**: PRE-REFACTOR (should move to `src/core/assembler.rs`)  
**Purpose**: ELF assembly traits and interfaces  
**Action**: Move to `src/core/assembler.rs`

### 📦 `src/x64_encoder.rs`
**Status**: PRE-REFACTOR (should move to `src/x64/encoder.rs`)  
**Purpose**: x86-64 instruction encoding using iced-x86  
**Action**: Move to `src/x64/encoder.rs`

### 📦 `src/x64_backend.rs`
**Status**: PRE-REFACTOR (should move to `src/x64/backend.rs`)  
**Purpose**: x86-64 backend implementation  
**Action**: Move to `src/x64/backend.rs`

### 📦 `src/calling_convention.rs`
**Status**: PRE-REFACTOR (should move to `src/x64/calling_convention.rs`)  
**Purpose**: System V calling convention implementation  
**Action**: Move to `src/x64/calling_convention.rs`

## Legacy Files (Should be Deprecated)

### ⚠️ `src/complete_compiler.rs`
**Status**: LEGACY (4000+ lines)  
**Purpose**: Generic trait-based compiler implementation  
**Action**: Move to `src/legacy/generic_compiler.rs`

### ⚠️ `src/llvm_adaptor/basic.rs`
**Status**: LEGACY  
**Purpose**: Minimal LLVM adaptor for simple cases  
**Action**: Move to `src/legacy/minimal_adaptor.rs`

### ⚠️ `src/function_analyzer.rs`
**Status**: LEGACY (superseded by arena version)  
**Purpose**: Original function analyzer  
**Action**: Remove or move to legacy

### ⚠️ `src/function_analyzer_v2.rs`
**Status**: LEGACY (experimental version)  
**Purpose**: Intermediate analyzer version  
**Action**: Remove

### ⚠️ `src/instruction_selection_test.rs`
**Status**: LEGACY  
**Purpose**: Test for old instruction selection approach  
**Action**: Update or remove

## Core Framework Files

### 🔧 `src/lib.rs`
**Status**: Updated in refactor  
**Purpose**: Main library entry point with public API  
**Action**: Continue cleanup of exports

### 🔧 `src/adaptor.rs`
**Status**: Core trait definition  
**Purpose**: IrAdaptor trait that all IR adaptors implement  
**Action**: Keep in place

### 🔧 `src/compiler.rs`
**Status**: Core trait definition  
**Purpose**: CompilerBase trait and Backend trait definitions  
**Action**: Keep in place

### 🔧 `src/analyzer.rs`
**Status**: Core component  
**Purpose**: Block ordering and liveness analysis  
**Action**: Keep in place

### 🔧 `src/value_ref.rs`
**Status**: Core component  
**Purpose**: Value reference and part management  
**Action**: Consider moving to `src/core/`

### 🔧 `src/function_codegen.rs`
**Status**: Core component  
**Purpose**: Function code generation infrastructure  
**Action**: Consider moving to `src/core/`

## Utility/Support Files

### 📦 `src/llvm_compiler.rs`
**Status**: Just traits and enums  
**Purpose**: LLVM-specific traits and instruction categories  
**Action**: Move content to `src/llvm/traits.rs`

### 📦 `src/llvm_adaptor.rs`
**Status**: Just module declarations  
**Purpose**: Module file for LLVM adaptors  
**Action**: Remove after moving submodules

### 📦 `src/function_analyzer_arena.rs`
**Status**: Best analyzer implementation  
**Purpose**: Arena-based function analysis  
**Action**: Move to `src/llvm/function_analysis.rs`

### 📦 `src/function_analysis_result.rs`
**Status**: Support type  
**Purpose**: Analysis result types  
**Action**: Move with function analyzer

### 📦 `src/encodegen.rs`
**Status**: Utility  
**Purpose**: LLVM IR parsing for code generation  
**Action**: Keep in place or move to tools/

### 📦 `src/bin/encodegen.rs`
**Status**: Binary  
**Purpose**: CLI tool for encoding snippet generation  
**Action**: Keep in place

## Documentation Files

### 🔧 `src/overview.rs`
**Status**: Documentation  
**Purpose**: Framework overview  
**Action**: Update to reflect new structure

### 🔧 `src/guide.rs`
**Status**: Documentation  
**Purpose**: Example IR guide  
**Action**: Update or move to docs/

## Summary of Required Actions

1. **Immediate moves** (preserving git history):
   - `llvm_compiler_concrete.rs` → `llvm/compiler.rs`
   - `llvm_adaptor/enhanced.rs` → `llvm/adaptor.rs`
   - `phi_resolver.rs` → `llvm/phi_resolution.rs`
   - Core infrastructure files to `core/` subdirectory
   - x64-specific files to `x64/` subdirectory

2. **Legacy moves**:
   - `complete_compiler.rs` → `legacy/generic_compiler.rs`
   - `llvm_adaptor/basic.rs` → `legacy/minimal_adaptor.rs`

3. **Cleanup**:
   - Remove empty module files after moves
   - Remove experimental analyzer versions
   - Update all imports and module declarations
   - Remove outdated comments and TODOs

4. **Documentation updates**:
   - Update overview.rs to reflect new architecture
   - Update README and other docs
   - Add module-level documentation to new mod.rs files