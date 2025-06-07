# TPDE Memory Management Strategy

## Overview

TPDE uses arena-based memory allocation as a core design principle. This strategy is specifically designed for embedding the compiler within a database system where memory usage must be strictly controlled and predictable.

## Database Embedding Context

In a database system:
- The majority of server RAM is dedicated to data pages, buffer pools, and caches
- Query compilation must use a small, bounded amount of memory
- Memory fragmentation from many small allocations degrades performance
- Each query compilation must have deterministic cleanup

## Arena Allocation Benefits

1. **Bounded Memory Usage**: Each compilation gets a fixed-size arena
2. **Zero Fragmentation**: No heap fragmentation from malloc/free cycles
3. **Fast Deallocation**: Drop arena = free all memory in one operation
4. **Memory Accounting**: Easy to track and limit memory per compilation

## Implementation Pattern

```rust
// Database integration example
pub struct QueryCompiler {
    memory_limit: usize,
}

impl QueryCompiler {
    pub fn compile_query(&self, query: &str) -> Result<CompiledQuery> {
        // Each query gets its own bounded arena
        let arena = Bump::with_capacity(self.memory_limit);
        let session = CompilationSession::with_memory_limit(&arena, self.memory_limit);
        
        // Compile with automatic memory tracking
        let result = compile_with_session(&session, query)?;
        
        // Check memory at major stages
        session.check_memory_limit()?;
        
        // Arena dropped here - all memory reclaimed
        Ok(result)
    }
}
```

## Lifetime Management

The `'arena` lifetime parameter throughout the codebase ensures:
- No allocation can escape the compilation session
- Compile-time prevention of use-after-free bugs
- Clear ownership and lifetime relationships

## Testing Strategy

Tests use `TestContext` to simulate database memory management:
```rust
with_test_context(|ctx| {
    let session = ctx.session();
    // Test code using arena-allocated objects
});
// Arena cleaned up automatically
```

## Memory Limits

Memory limits are enforced at major compilation stages:
- After IR analysis
- After register allocation  
- After code generation
- Before final assembly

This fail-fast approach ensures runaway compilations are caught early.