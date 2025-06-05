//! Arena-based compilation session management.
//!
//! This module provides simplified lifetime management for TPDE compilation
//! using arena allocation. All compilation objects are tied to the session
//! lifetime, eliminating complex lifetime propagation.

use bumpalo::Bump;
use std::fmt;

/// Arena-based compilation session.
///
/// This manages the lifetime of all compilation objects, using arena allocation
/// to simplify memory management. All compilation data structures are allocated
/// in the arena and have the same lifetime as the compilation session.
pub struct CompilationSession<'arena> {
    /// Arena allocator for compilation objects.
    arena: &'arena Bump,
    
    /// Session statistics for debugging and optimization.
    stats: SessionStats,
}

impl<'arena> CompilationSession<'arena> {
    /// Create a new compilation session with the given arena.
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            stats: SessionStats::default(),
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
        T: Clone 
    {
        self.arena.alloc_slice_clone(slice)
    }
    
    /// Record that a function was compiled.
    pub fn record_function_compiled(&mut self, name: &str, code_size: usize) {
        self.stats.functions_compiled += 1;
        self.stats.total_code_size += code_size;
        
        if self.stats.largest_function_size < code_size {
            self.stats.largest_function_size = code_size;
            self.stats.largest_function_name = name.to_string();
        }
    }
    
    /// Record an instruction compilation.
    pub fn record_instruction_compiled(&mut self, opcode: &str) {
        self.stats.instructions_compiled += 1;
        *self.stats.instruction_counts.entry(opcode.to_string()).or_insert(0) += 1;
    }
    
    /// Get compilation statistics.
    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }
}

/// Compilation session statistics.
#[derive(Debug, Default)]
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
}

impl fmt::Display for SessionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Compilation Session Statistics:")?;
        writeln!(f, "  Functions compiled: {}", self.functions_compiled)?;
        writeln!(f, "  Instructions compiled: {}", self.instructions_compiled)?;
        writeln!(f, "  Total code size: {} bytes", self.total_code_size)?;
        
        if !self.largest_function_name.is_empty() {
            writeln!(f, "  Largest function: {} ({} bytes)", 
                    self.largest_function_name, self.largest_function_size)?;
        }
        
        if !self.instruction_counts.is_empty() {
            writeln!(f, "  Instruction breakdown:")?;
            let mut sorted: Vec<_> = self.instruction_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
            
            for (opcode, count) in sorted.iter().take(10) { // Top 10
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
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::AllocationFailed => write!(f, "Arena allocation failed"),
            SessionError::InvalidState(msg) => write!(f, "Invalid session state: {}", msg),
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
        
        assert_eq!(session.stats().functions_compiled, 0);
        assert_eq!(session.stats().instructions_compiled, 0);
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
    fn test_session_statistics() {
        let arena = Bump::new();
        let mut session = CompilationSession::new(&arena);
        
        session.record_function_compiled("test_func", 128);
        session.record_instruction_compiled("add");
        session.record_instruction_compiled("icmp");
        session.record_instruction_compiled("add");
        
        let stats = session.stats();
        assert_eq!(stats.functions_compiled, 1);
        assert_eq!(stats.instructions_compiled, 3);
        assert_eq!(stats.total_code_size, 128);
        assert_eq!(stats.instruction_counts["add"], 2);
        assert_eq!(stats.instruction_counts["icmp"], 1);
    }
    
    #[test]
    fn test_statistics_display() {
        let arena = Bump::new();
        let mut session = CompilationSession::new(&arena);
        
        session.record_function_compiled("factorial", 256);
        session.record_instruction_compiled("add");
        session.record_instruction_compiled("icmp");
        
        let output = format!("{}", session.stats());
        assert!(output.contains("Functions compiled: 1"));
        assert!(output.contains("Instructions compiled: 2"));
        assert!(output.contains("factorial (256 bytes)"));
    }
}