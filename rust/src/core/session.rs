// This module provides arena-based compilation session management using the bumpalo crate
// to simplify lifetime management in TPDE. CompilationSession is the central hub that owns
// the arena allocator and tracks all compilation state with a unified lifetime. It stores
// value locations (register/stack/memory), block information (layout order, predecessors,
// successors, PHI nodes), PHI node resolution data, interned strings, and compilation
// statistics. All objects are allocated in the arena and share the session lifetime,
// eliminating complex lifetime annotations. The session provides methods to track function
// compilation progress, record instruction counts, manage PHI nodes, and gather performance
// metrics. SessionStats tracks compilation metrics like function count, code size, instruction
// breakdown, register allocations, and spills. This arena-based approach was introduced during
// the architectural redesign to replace generic trait bounds with concrete LLVM types.

//! Arena-based compilation session management.
//!
//! This module provides simplified lifetime management for TPDE compilation
//! using arena allocation. All compilation objects are tied to the session
//! lifetime, eliminating complex lifetime propagation.

use bumpalo::Bump;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

// LLVM types not needed for the arena-based session management

/// Value location information stored in the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueLocation {
    Register(u8),
    Stack(i32),
    Constant(i64),
    Memory { base: u8, offset: i32 },
}

/// Block information tracked during compilation.
#[derive(Debug, Clone)]
pub struct BlockInfo {
    pub layout_order: usize,
    pub is_visited: bool,
    pub predecessors: Vec<usize>, // Block indices
    pub successors: Vec<usize>,   // Block indices
    pub phi_nodes: Vec<usize>,    // Instruction indices
}

/// PHI node information for resolution.
#[derive(Debug, Clone)]
pub struct PhiNodeInfo {
    pub result_value: usize,
    pub incoming_values: Vec<(usize, usize)>, // (value_idx, block_idx)
}

/// Arena-based compilation session.
///
/// This manages the lifetime of all compilation objects, using arena allocation
/// to simplify memory management. All compilation data structures are allocated
/// in the arena and have the same lifetime as the compilation session.
pub struct CompilationSession<'arena> {
    /// Arena allocator for compilation objects.
    arena: &'arena Bump,

    /// Session statistics for debugging and optimization.
    stats: RefCell<SessionStats>,

    /// Value locations tracked during compilation.
    value_locations: RefCell<HashMap<usize, ValueLocation>>,

    /// Block information for control flow.
    block_info: RefCell<HashMap<usize, BlockInfo>>,

    /// PHI node information for resolution.
    phi_nodes: RefCell<HashMap<usize, PhiNodeInfo>>,

    /// String interning for efficient storage.
    interned_strings: RefCell<HashMap<String, &'arena str>>,

    /// Current function being compiled.
    current_function: RefCell<Option<String>>,
}

impl<'arena> CompilationSession<'arena> {
    /// Create a new compilation session with the given arena.
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            stats: RefCell::new(SessionStats::default()),
            value_locations: RefCell::new(HashMap::new()),
            block_info: RefCell::new(HashMap::new()),
            phi_nodes: RefCell::new(HashMap::new()),
            interned_strings: RefCell::new(HashMap::new()),
            current_function: RefCell::new(None),
        }
    }

    /// Get access to the arena allocator.
    pub fn arena(&self) -> &'arena Bump {
        self.arena
    }

    /// Allocate an object in the session arena.
    pub fn alloc<T>(&self, value: T) -> &'arena mut T {
        self.arena.alloc(value)
    }

    /// Allocate a slice in the session arena.
    pub fn alloc_slice<T>(&self, slice: &[T]) -> &'arena [T]
    where
        T: Clone,
    {
        self.arena.alloc_slice_clone(slice)
    }

    /// Intern a string in the arena.
    pub fn intern_str(&self, s: &str) -> &'arena str {
        let mut strings = self.interned_strings.borrow_mut();
        if let Some(&interned) = strings.get(s) {
            return interned;
        }

        let interned = self.arena.alloc_str(s);
        strings.insert(s.to_string(), interned);
        interned
    }

    /// Set current function being compiled.
    pub fn set_current_function(&self, name: &str) {
        *self.current_function.borrow_mut() = Some(name.to_string());
    }

    /// Track value location.
    pub fn set_value_location(&self, value_idx: usize, location: ValueLocation) {
        self.value_locations
            .borrow_mut()
            .insert(value_idx, location);
    }

    /// Get value location.
    pub fn get_value_location(&self, value_idx: usize) -> Option<ValueLocation> {
        self.value_locations.borrow().get(&value_idx).copied()
    }

    /// Add block information.
    pub fn add_block_info(&self, block_idx: usize, info: BlockInfo) {
        self.block_info.borrow_mut().insert(block_idx, info);
    }

    /// Get block information.
    pub fn get_block_info(&self, block_idx: usize) -> Option<BlockInfo> {
        self.block_info.borrow().get(&block_idx).cloned()
    }

    /// Add PHI node information.
    pub fn add_phi_node(&self, inst_idx: usize, info: PhiNodeInfo) {
        self.phi_nodes.borrow_mut().insert(inst_idx, info);
    }

    /// Get PHI node information.
    pub fn get_phi_node(&self, inst_idx: usize) -> Option<PhiNodeInfo> {
        self.phi_nodes.borrow().get(&inst_idx).cloned()
    }

    /// Get all PHI nodes for resolution.
    pub fn get_all_phi_nodes(&self) -> Vec<(usize, PhiNodeInfo)> {
        self.phi_nodes
            .borrow()
            .iter()
            .map(|(idx, info)| (*idx, info.clone()))
            .collect()
    }

    /// Clear session state for new function.
    pub fn clear_function_state(&self) {
        self.value_locations.borrow_mut().clear();
        self.block_info.borrow_mut().clear();
        self.phi_nodes.borrow_mut().clear();
        *self.current_function.borrow_mut() = None;
    }

    /// Record that a function was compiled.
    pub fn record_function_compiled(&self, name: &str, code_size: usize) {
        let mut stats = self.stats.borrow_mut();
        stats.functions_compiled += 1;
        stats.total_code_size += code_size;

        if stats.largest_function_size < code_size {
            stats.largest_function_size = code_size;
            stats.largest_function_name = name.to_string();
        }
    }

    /// Record an instruction compilation.
    pub fn record_instruction_compiled(&self, opcode: &str) {
        let mut stats = self.stats.borrow_mut();
        stats.instructions_compiled += 1;
        *stats
            .instruction_counts
            .entry(opcode.to_string())
            .or_insert(0) += 1;
    }

    /// Record PHI node resolution.
    pub fn record_phi_resolved(&self) {
        self.stats.borrow_mut().phi_nodes_resolved += 1;
    }

    /// Record register allocation.
    pub fn record_register_allocated(&self) {
        self.stats.borrow_mut().registers_allocated += 1;
    }

    /// Record spill generation.
    pub fn record_spill_generated(&self) {
        self.stats.borrow_mut().spills_generated += 1;
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> SessionStats {
        self.stats.borrow().clone()
    }

    /// Record a call site for later relocation.
    pub fn record_call_site(&self, function_name: String) {
        self.stats.borrow_mut().total_calls += 1;
        // In a real implementation, we would store the call site location
        // and function name for the linker to resolve
        log::debug!("Call site recorded: {}", function_name);
    }
}

/// Compilation session statistics.
#[derive(Debug, Default, Clone)]
pub struct SessionStats {
    /// Number of functions compiled.
    pub functions_compiled: usize,

    /// Total code size generated (bytes).
    pub total_code_size: usize,

    /// Number of instructions compiled.
    pub instructions_compiled: usize,

    /// Count of each instruction type compiled.
    pub instruction_counts: std::collections::HashMap<String, usize>,

    /// Largest function compiled (for analysis).
    pub largest_function_size: usize,

    /// Name of largest function.
    pub largest_function_name: String,

    /// PHI nodes resolved.
    pub phi_nodes_resolved: usize,

    /// Registers allocated.
    pub registers_allocated: usize,

    /// Spills generated.
    pub spills_generated: usize,

    /// Total function calls compiled.
    pub total_calls: usize,
}

impl fmt::Display for SessionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Compilation Session Statistics:")?;
        writeln!(f, "  Functions compiled: {}", self.functions_compiled)?;
        writeln!(f, "  Instructions compiled: {}", self.instructions_compiled)?;
        writeln!(f, "  Total code size: {} bytes", self.total_code_size)?;
        writeln!(f, "  PHI nodes resolved: {}", self.phi_nodes_resolved)?;
        writeln!(f, "  Registers allocated: {}", self.registers_allocated)?;
        writeln!(f, "  Spills generated: {}", self.spills_generated)?;
        writeln!(f, "  Function calls compiled: {}", self.total_calls)?;

        if !self.largest_function_name.is_empty() {
            writeln!(
                f,
                "  Largest function: {} ({} bytes)",
                self.largest_function_name, self.largest_function_size
            )?;
        }

        if !self.instruction_counts.is_empty() {
            writeln!(f, "  Instruction breakdown:")?;
            let mut sorted: Vec<_> = self.instruction_counts.iter().collect();
            sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

            for (opcode, count) in sorted.into_iter().take(10) {
                writeln!(f, "    {}: {}", opcode, count)?;
            }
        }

        Ok(())
    }
}

/// Error types for compilation session operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Arena allocation failed.
    AllocationFailed,

    /// Invalid session state.
    InvalidState(String),

    /// Function not found.
    FunctionNotFound(String),

    /// Block not found.
    BlockNotFound(usize),

    /// Value not found.
    ValueNotFound(usize),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::AllocationFailed => write!(f, "Arena allocation failed"),
            SessionError::InvalidState(msg) => write!(f, "Invalid session state: {}", msg),
            SessionError::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            SessionError::BlockNotFound(idx) => write!(f, "Block not found: {}", idx),
            SessionError::ValueNotFound(idx) => write!(f, "Value not found: {}", idx),
        }
    }
}

impl std::error::Error for SessionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_session_creation() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let stats = session.stats();
        assert_eq!(stats.functions_compiled, 0);
        assert_eq!(stats.instructions_compiled, 0);
        assert_eq!(stats.phi_nodes_resolved, 0);
    }

    #[test]
    fn test_arena_allocation() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let value = session.alloc(42);
        assert_eq!(*value, 42);

        let slice = session.alloc_slice(&[1, 2, 3, 4]);
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_value_tracking() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        session.set_value_location(0, ValueLocation::Register(3));
        session.set_value_location(1, ValueLocation::Stack(-8));

        assert_eq!(
            session.get_value_location(0),
            Some(ValueLocation::Register(3))
        );
        assert_eq!(
            session.get_value_location(1),
            Some(ValueLocation::Stack(-8))
        );
        assert_eq!(session.get_value_location(2), None);
    }

    #[test]
    fn test_string_interning() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let s1 = session.intern_str("hello");
        let s2 = session.intern_str("hello");
        let s3 = session.intern_str("world");

        assert_eq!(s1.as_ptr(), s2.as_ptr()); // Same string interned
        assert_ne!(s1.as_ptr(), s3.as_ptr()); // Different strings
    }

    #[test]
    fn test_session_statistics() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        session.record_function_compiled("test_func", 128);
        session.record_instruction_compiled("add");
        session.record_instruction_compiled("icmp");
        session.record_instruction_compiled("add");
        session.record_phi_resolved();
        session.record_register_allocated();
        session.record_register_allocated();
        session.record_spill_generated();

        let stats = session.stats();
        assert_eq!(stats.functions_compiled, 1);
        assert_eq!(stats.instructions_compiled, 3);
        assert_eq!(stats.total_code_size, 128);
        assert_eq!(stats.instruction_counts["add"], 2);
        assert_eq!(stats.instruction_counts["icmp"], 1);
        assert_eq!(stats.phi_nodes_resolved, 1);
        assert_eq!(stats.registers_allocated, 2);
        assert_eq!(stats.spills_generated, 1);
    }

    #[test]
    fn test_block_info() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let info = BlockInfo {
            layout_order: 0,
            is_visited: false,
            predecessors: vec![],
            successors: vec![1, 2],
            phi_nodes: vec![],
        };

        session.add_block_info(0, info.clone());
        let retrieved = session.get_block_info(0).unwrap();
        assert_eq!(retrieved.successors, vec![1, 2]);
    }

    #[test]
    fn test_phi_node_tracking() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let phi = PhiNodeInfo {
            result_value: 10,
            incoming_values: vec![(5, 0), (6, 1)],
        };

        session.add_phi_node(100, phi.clone());
        let retrieved = session.get_phi_node(100).unwrap();
        assert_eq!(retrieved.result_value, 10);
        assert_eq!(retrieved.incoming_values.len(), 2);

        let all_phis = session.get_all_phi_nodes();
        assert_eq!(all_phis.len(), 1);
        assert_eq!(all_phis[0].0, 100);
    }

    #[test]
    fn test_statistics_display() {
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        session.record_function_compiled("factorial", 256);
        session.record_instruction_compiled("add");
        session.record_instruction_compiled("icmp");

        let output = format!("{}", session.stats());
        assert!(output.contains("Functions compiled: 1"));
        assert!(output.contains("Instructions compiled: 2"));
        assert!(output.contains("factorial (256 bytes)"));
    }
}
