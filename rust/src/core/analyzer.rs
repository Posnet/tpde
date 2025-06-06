// This module implements the Analyzer component that computes block layout and value liveness
// information for SSA IR functions. It performs two key analyses: 1) Reverse Post-Order (RPO)
// traversal to determine optimal block ordering for code generation, and 2) Liveness analysis
// to track when values are first defined and last used across basic blocks. The analyzer
// works with any IR through the IrAdaptor trait, building a block ordering map and tracking
// LivenessInfo (first/last block indices, reference count, and whether values must stay live
// through entire blocks). This information is critical for the register allocator to make
// efficient allocation decisions and minimize spills. The implementation follows the C++
// version's algorithm, using depth-first search for RPO computation and linear scanning
// for liveness tracking.

use super::adaptor::IrAdaptor;
use core::marker::PhantomData;
use std::collections::{HashMap, HashSet};

/// Liveness information for a single value.
#[derive(Default, Clone, Copy)]
pub struct LivenessInfo {
    /// Index of the first block this value is live in.
    pub first: usize,
    /// Index of the last block this value is live in.
    pub last: usize,
    /// Number of uses including the definition.
    pub ref_count: u32,
    /// Whether the value must stay allocated until the end of `last`.
    pub last_full: bool,
}

/// Computes block layout and liveness information for a function.
///
/// The analyzer walks the IR provided by [`IrAdaptor`] in a reverse
/// post-order and records begin/end positions for each value.  This data drives
/// the register allocator within [`CompilerBase`].  The algorithm closely
/// follows the description in the C++ docs and is summarized in
/// [`overview`].
#[allow(dead_code)]
pub struct Analyzer<A: IrAdaptor> {
    order: Vec<A::BlockRef>,
    block_map: HashMap<A::BlockRef, usize>,
    liveness: Vec<LivenessInfo>,
    _marker: PhantomData<A>,
}

impl<A: IrAdaptor> Default for Analyzer<A> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            block_map: HashMap::new(),
            liveness: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<A: IrAdaptor> Analyzer<A> {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sequence of blocks in reverse post order.
    pub fn order(&self) -> &[A::BlockRef] {
        &self.order
    }

    /// Liveness information for a value index.
    pub fn liveness(&self, idx: usize) -> Option<&LivenessInfo> {
        self.liveness.get(idx)
    }

    /// Build block layout and liveness for the given function using the adaptor.
    pub fn switch_func(&mut self, adaptor: &mut A, func: A::FuncRef) {
        self.order.clear();
        self.block_map.clear();
        self.liveness.clear();

        if !adaptor.switch_func(func) {
            return;
        }

        // -------- build RPO order ---------
        let entry = adaptor.entry_block();
        let mut post = Vec::new();
        let mut stack = vec![(entry, false)];
        let mut visited = HashSet::new();
        while let Some((block, processed)) = stack.pop() {
            if processed {
                post.push(block);
                continue;
            }
            if !visited.insert(block) {
                continue;
            }
            stack.push((block, true));
            for succ in adaptor.block_succs(block) {
                stack.push((succ, false));
            }
        }
        post.reverse();
        self.order = post;
        for (idx, b) in self.order.iter().enumerate() {
            self.block_map.insert(*b, idx);
        }

        // -------- compute liveness ---------
        for idx in 0..self.order.len() {
            let block = self.order[idx];
            for inst in adaptor.block_insts(block) {
                for val in adaptor.inst_results(inst) {
                    self.record(adaptor, val, idx);
                }
                for val in adaptor.inst_operands(inst) {
                    self.record(adaptor, val, idx);
                }
            }
        }
    }

    fn record(&mut self, adaptor: &A, val: A::ValueRef, block_idx: usize) {
        if adaptor.val_ignore_liveness(val) {
            return;
        }
        let idx = adaptor.val_local_idx(val);
        if idx >= self.liveness.len() {
            self.liveness.resize(
                idx + 1,
                LivenessInfo { first: block_idx, last: block_idx, ref_count: 0, last_full: false },
            );
        }
        let info = &mut self.liveness[idx];
        info.ref_count += 1;
        if info.ref_count == 1 {
            info.first = block_idx;
            info.last = block_idx;
        } else {
            if block_idx < info.first {
                info.first = block_idx;
            }
            if block_idx > info.last {
                info.last = block_idx;
            }
        }
        info.last_full = info.first != info.last;
    }
}
