//! Test utilities for arena-based testing.
//!
//! This module provides utilities for testing arena-allocated code without
//! memory leaks, simulating how the compiler will be embedded in a database.

#[cfg(test)]
pub mod test {
    use super::super::session::CompilationSession;
    use bumpalo::Bump;

    /// Test context that manages arena lifetime for tests.
    ///
    /// This simulates how a database would manage compilation memory,
    /// providing a bounded arena for each compilation session.
    pub struct TestContext {
        arena: Bump,
    }

    impl TestContext {
        /// Create a new test context with default capacity.
        pub fn new() -> Self {
            Self { arena: Bump::new() }
        }

        /// Create a test context with specified capacity, simulating memory limits.
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                arena: Bump::with_capacity(capacity),
            }
        }

        /// Get a reference to the arena.
        pub fn arena(&self) -> &Bump {
            &self.arena
        }

        /// Create a compilation session using this context's arena.
        ///
        /// The returned session borrows from the TestContext, so the
        /// TestContext must outlive the session.
        pub fn create_session(&self) -> CompilationSession<'_> {
            CompilationSession::new(&self.arena)
        }

        /// Get current memory usage for testing memory limits.
        pub fn memory_used(&self) -> usize {
            self.arena.allocated_bytes()
        }

        /// Run a test with a session, ensuring proper cleanup.
        pub fn with_session<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&CompilationSession<'_>) -> R,
        {
            let session = self.create_session();
            f(&session)
        }
    }

    impl Default for TestContext {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Run a test with a temporary arena context.
    ///
    /// This is the preferred way to write tests that need arena allocation,
    /// as it ensures proper cleanup without leaks.
    pub fn with_test_context<F, R>(f: F) -> R
    where
        F: FnOnce(&TestContext) -> R,
    {
        let ctx = TestContext::new();
        f(&ctx)
    }

    /// Run a test with a memory-limited arena context.
    pub fn with_limited_context<F, R>(capacity: usize, f: F) -> R
    where
        F: FnOnce(&TestContext) -> R,
    {
        let ctx = TestContext::with_capacity(capacity);
        f(&ctx)
    }
}
