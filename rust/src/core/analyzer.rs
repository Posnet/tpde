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

/// Loop information for the analyzer.
#[derive(Clone, Copy, Debug)]
pub struct Loop {
    /// Nesting level (0 for root loop)
    pub level: u32,
    /// Parent loop index
    pub parent: u32,
    /// Beginning block index (inclusive)
    pub begin: u32,
    /// Ending block index (exclusive)
    pub end: u32,
    /// Number of blocks in this loop (including nested loops)
    pub num_blocks: u32,
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
    block_layout: Vec<A::BlockRef>,
    loops: Vec<Loop>,
    block_loop_map: Vec<u32>,
    _marker: PhantomData<A>,
}

impl<A: IrAdaptor> Default for Analyzer<A> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            block_map: HashMap::new(),
            liveness: Vec::new(),
            block_layout: Vec::new(),
            loops: Vec::new(),
            block_loop_map: Vec::new(),
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

    /// Get the block layout (blocks ordered for code generation).
    pub fn block_layout(&self) -> &[A::BlockRef] {
        &self.block_layout
    }

    /// Get loop information.
    pub fn loops(&self) -> &[Loop] {
        &self.loops
    }

    /// Build block layout and liveness for the given function using the adaptor.
    pub fn switch_func(&mut self, adaptor: &mut A, func: A::FuncRef) {
        self.order.clear();
        self.block_map.clear();
        self.liveness.clear();
        self.block_layout.clear();
        self.loops.clear();
        self.block_loop_map.clear();

        if !adaptor.switch_func(func) {
            return;
        }

        // Build block layout (which includes RPO and loop detection)
        self.build_block_layout(adaptor);

        // -------- compute liveness ---------
        // Visit function arguments if the adaptor wants us to
        // Note: This just marks them as defined in the entry block, it doesn't count as a use
        if A::TPDE_LIVENESS_VISIT_ARGS {
            for arg in adaptor.cur_args() {
                self.record(adaptor, arg, 0); // Entry block is always first in RPO
            }
        }

        for idx in 0..self.order.len() {
            let block = self.order[idx];

            // Process PHI nodes first
            for phi in adaptor.block_phis(block) {
                // Handle PHI operands specially - they're used in their incoming blocks
                let incoming_count = adaptor.phi_incoming_count(phi);
                for slot in 0..incoming_count {
                    let incoming_block = adaptor.phi_incoming_block_for_slot(phi, slot);
                    let incoming_value = adaptor.phi_incoming_val_for_slot(phi, slot);

                    // Find the index of the incoming block in our RPO order
                    if let Some(&incoming_idx) = self.block_map.get(&incoming_block) {
                        // Mark the incoming value as used in the incoming block
                        self.record(adaptor, incoming_value, incoming_idx);
                        // Also mark the PHI itself as used in the incoming block
                        self.record(adaptor, phi, incoming_idx);
                    }
                }
            }

            // Process regular instructions
            for inst in adaptor.block_insts(block) {
                for val in adaptor.inst_results(inst) {
                    self.record(adaptor, val, idx);
                }
                for val in adaptor.inst_operands(inst) {
                    self.record(adaptor, val, idx);
                }
            }
        }

        // Post-process to set last_full correctly
        // For now, use a simple heuristic: if a value spans multiple blocks
        // and the last block has successors (not terminating), set last_full = true
        for idx in 0..self.order.len() {
            let block = self.order[idx];
            let has_successors = adaptor.block_succs(block).count() > 0;

            // Update last_full for values whose last block is this one
            for info in &mut self.liveness {
                if info.last == idx && info.first != info.last && has_successors {
                    info.last_full = true;
                }
            }
        }
    }

    fn record(&mut self, adaptor: &A, val: A::ValueRef, block_idx: usize) {
        if adaptor.val_ignore_liveness(val) {
            return;
        }
        let idx = adaptor.val_local_idx(val);
        // Debug logging
        // eprintln!("Recording value with local_idx {} in block {}", idx, block_idx);
        if idx >= self.liveness.len() {
            self.liveness.resize(
                idx + 1,
                LivenessInfo {
                    first: block_idx,
                    last: block_idx,
                    ref_count: 0,
                    last_full: false,
                },
            );
        }
        let info = &mut self.liveness[idx];
        if info.ref_count == 0 {
            // First time seeing this value - initialize with ref_count 1
            info.ref_count = 1;
            info.first = block_idx;
            info.last = block_idx;
        } else {
            info.ref_count += 1;
            if block_idx < info.first {
                info.first = block_idx;
            }
            if block_idx > info.last {
                info.last = block_idx;
            }
        }
        // last_full should be false for now - it will be updated based on block type
        // when we have proper loop analysis
        info.last_full = false;
    }

    /// Build the complete block layout including loop detection.
    fn build_block_layout(&mut self, adaptor: &A) {
        // Build RPO order
        let rpo = self.build_rpo_block_order(adaptor);

        // Identify loops
        let (loop_parent, loop_heads) = self.identify_loops(adaptor, &rpo);

        // Build loop tree and final block layout
        self.build_loop_tree_and_block_layout(adaptor, rpo, loop_parent, loop_heads);
    }

    /// Build reverse post-order of blocks.
    fn build_rpo_block_order(&mut self, adaptor: &A) -> Vec<A::BlockRef> {
        let entry = adaptor.entry_block();

        // First pass: build a map of blocks to their order in the IR
        // This is used to sort successors to maintain source order
        let mut block_order_map = HashMap::new();
        for (idx, block) in adaptor.blocks().enumerate() {
            block_order_map.insert(block, idx as u32);
        }

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

            // Push successors onto the stack first, then sort them
            // This matches the C++ implementation exactly
            let start_idx = stack.len();
            for succ in adaptor.block_succs(block) {
                stack.push((succ, false));
            }

            // Sort the pushed children by their original block index
            // This ensures blocks appearing earlier in the IR are visited in the correct order
            let len = stack.len() - start_idx;
            if len > 1 {
                // Sort the portion of the stack we just added
                let slice = &mut stack[start_idx..];
                slice.sort_by_key(|(block, _)| {
                    block_order_map.get(block).copied().unwrap_or(u32::MAX)
                });
            }
        }

        post.reverse();
        self.order = post.clone();
        for (idx, b) in self.order.iter().enumerate() {
            self.block_map.insert(*b, idx);
        }

        post
    }

    /// Identify loops using the algorithm from Wei et al.
    fn identify_loops(&self, adaptor: &A, block_rpo: &[A::BlockRef]) -> (Vec<u32>, HashSet<usize>) {
        let mut loop_parent = vec![0; block_rpo.len()];
        let mut loop_heads = HashSet::new();

        #[derive(Default, Clone)]
        struct BlockInfo {
            traversed: bool,
            self_loop: bool,
            dfsp_pos: u32,
            iloop_header: u32,
        }

        let mut block_infos = vec![BlockInfo::default(); block_rpo.len()];

        // Helper function to tag loop headers
        let tag_lhead = |block_infos: &mut Vec<BlockInfo>, b: u32, h: u32| {
            if b == h || h == 0 {
                return;
            }

            let mut cur1 = b;
            let mut cur2 = h;
            while block_infos[cur1 as usize].iloop_header != 0 {
                let ih = block_infos[cur1 as usize].iloop_header;
                if ih == cur2 {
                    return;
                }
                if block_infos[ih as usize].dfsp_pos < block_infos[cur2 as usize].dfsp_pos {
                    block_infos[cur1 as usize].iloop_header = cur2;
                    cur1 = cur2;
                    cur2 = ih;
                } else {
                    cur1 = ih;
                }
            }
            block_infos[cur1 as usize].iloop_header = cur2;
        };

        // Stack entry for DFS
        enum StackState {
            Visit {
                block_idx: usize,
                dfsp_pos: u32,
            },
            PostProcess {
                block_idx: usize,
                parent_idx: Option<usize>,
            },
        }

        // DFS to identify loops
        let mut stack = vec![StackState::Visit {
            block_idx: 0,
            dfsp_pos: 1,
        }];
        let mut dfsp_counter = 1u32;

        while let Some(state) = stack.pop() {
            match state {
                StackState::Visit {
                    block_idx,
                    dfsp_pos,
                } => {
                    if block_infos[block_idx].traversed {
                        continue;
                    }

                    block_infos[block_idx].traversed = true;
                    block_infos[block_idx].dfsp_pos = dfsp_pos;

                    // Push post-process for after children
                    stack.push(StackState::PostProcess {
                        block_idx,
                        parent_idx: None,
                    });

                    // Process successors in reverse order (they'll be popped in correct order)
                    let succs: Vec<_> = adaptor
                        .block_succs(block_rpo[block_idx])
                        .filter_map(|succ| self.block_map.get(&succ).copied())
                        .collect();

                    for &succ_idx in succs.iter().rev() {
                        if succ_idx == block_idx {
                            block_infos[block_idx].self_loop = true;
                        }

                        if !block_infos[succ_idx].traversed {
                            // Need to visit this successor
                            dfsp_counter += 1;
                            stack.push(StackState::PostProcess {
                                block_idx: succ_idx,
                                parent_idx: Some(block_idx),
                            });
                            stack.push(StackState::Visit {
                                block_idx: succ_idx,
                                dfsp_pos: dfsp_counter,
                            });
                        } else if block_infos[succ_idx].dfsp_pos > 0 {
                            // Back edge
                            tag_lhead(&mut block_infos, block_idx as u32, succ_idx as u32);
                        } else if block_infos[succ_idx].iloop_header != 0 {
                            // Forward edge to a block with a loop header
                            let mut h_idx = block_infos[succ_idx].iloop_header;
                            if block_infos[h_idx as usize].dfsp_pos > 0 {
                                tag_lhead(&mut block_infos, block_idx as u32, h_idx);
                            } else {
                                while block_infos[h_idx as usize].iloop_header != 0 {
                                    h_idx = block_infos[h_idx as usize].iloop_header;
                                    if block_infos[h_idx as usize].dfsp_pos > 0 {
                                        tag_lhead(&mut block_infos, block_idx as u32, h_idx);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                StackState::PostProcess {
                    block_idx,
                    parent_idx,
                } => {
                    // Mark as post-processed
                    block_infos[block_idx].dfsp_pos = 0;

                    // If we have a parent, tag it with our loop header
                    if let Some(parent) = parent_idx {
                        let nh = block_infos[block_idx].iloop_header;
                        tag_lhead(&mut block_infos, parent as u32, nh);
                    }
                }
            }
        }

        // Convert results
        for i in 0..block_rpo.len() {
            let info = &block_infos[i];
            if info.iloop_header != 0 {
                loop_parent[i] = info.iloop_header;
                loop_heads.insert(info.iloop_header as usize);
            }
            if info.self_loop {
                loop_heads.insert(i);
            }
        }

        // Entry block is always a loop head
        loop_heads.insert(0);

        (loop_parent, loop_heads)
    }

    /// Build the loop tree and final block layout.
    fn build_loop_tree_and_block_layout(
        &mut self,
        adaptor: &A,
        block_rpo: Vec<A::BlockRef>,
        loop_parent: Vec<u32>,
        loop_heads: HashSet<usize>,
    ) {
        #[derive(Clone, Copy)]
        struct BlockLoopInfo {
            loop_idx: u32,
            rpo_idx: u32,
        }

        let mut loop_blocks = vec![
            BlockLoopInfo {
                loop_idx: !0,
                rpo_idx: 0
            };
            block_rpo.len()
        ];
        for (i, block_info) in loop_blocks.iter_mut().enumerate() {
            block_info.rpo_idx = i as u32;
        }

        // Initialize loops with root loop
        self.loops.clear();
        self.loops.push(Loop {
            level: 0,
            parent: 0,
            begin: 0,
            end: 0,
            num_blocks: 1,
        });
        loop_blocks[0].loop_idx = 0;

        // Helper to build or get parent loop
        fn build_or_get_parent_loop(
            loops: &mut Vec<Loop>,
            loop_blocks: &mut [BlockLoopInfo],
            loop_parent: &[u32],
            i: usize,
        ) -> u32 {
            let parent = loop_parent[i] as usize;
            if loop_blocks[parent].loop_idx != !0 {
                // Already have a loop for this block
                loop_blocks[parent].loop_idx
            } else {
                // Recursively get parent loop
                let parent_loop_idx =
                    build_or_get_parent_loop(loops, loop_blocks, loop_parent, parent);
                let loop_idx = loops.len() as u32;
                loops.push(Loop {
                    level: loops[parent_loop_idx as usize].level + 1,
                    parent: parent_loop_idx,
                    begin: 0,
                    end: 0,
                    num_blocks: 0,
                });
                loop_blocks[parent].loop_idx = loop_idx;
                loop_idx
            }
        }

        // Build loop tree
        for i in 1..loop_parent.len() {
            let parent_loop =
                build_or_get_parent_loop(&mut self.loops, &mut loop_blocks, &loop_parent, i);

            if loop_heads.contains(&i) {
                // This is a loop head
                let mut loop_idx = loop_blocks[i].loop_idx;
                if loop_idx == !0 {
                    // Create new loop
                    loop_idx = self.loops.len() as u32;
                    self.loops.push(Loop {
                        level: self.loops[parent_loop as usize].level + 1,
                        parent: parent_loop,
                        begin: 0,
                        end: 0,
                        num_blocks: 0,
                    });
                    loop_blocks[i].loop_idx = loop_idx;
                }
                self.loops[loop_idx as usize].num_blocks += 1;
            } else {
                loop_blocks[i].loop_idx = parent_loop;
                self.loops[parent_loop as usize].num_blocks += 1;
            }
        }

        // Accumulate total blocks in each loop
        for i in (1..self.loops.len()).rev() {
            let parent = self.loops[i].parent as usize;
            self.loops[parent].num_blocks += self.loops[i].num_blocks;
        }

        // Layout blocks
        self.block_layout.clear();
        self.block_layout.resize(block_rpo.len(), block_rpo[0]); // Placeholder
        self.block_loop_map.clear();
        self.block_loop_map.resize(block_rpo.len(), 0);

        // Initialize root loop
        self.loops[0].begin = 0;
        self.loops[0].end = 0;

        // Layout loops
        let mut loop_begins = vec![None; self.loops.len()];

        for i in 0..block_rpo.len() {
            let loop_idx = loop_blocks[i].loop_idx as usize;

            // Ensure loop has been laid out
            if loop_begins[loop_idx].is_none() {
                // Layout this loop
                let mut curr_loop = loop_idx;
                let mut to_layout = vec![];

                // Find all loops that need to be laid out
                while loop_begins[curr_loop].is_none() && curr_loop != 0 {
                    to_layout.push(curr_loop);
                    curr_loop = self.loops[curr_loop].parent as usize;
                }

                // Layout in reverse order (parent to child)
                for &lidx in to_layout.iter().rev() {
                    let parent = self.loops[lidx].parent as usize;
                    let begin = self.loops[parent].end;
                    self.loops[parent].end = begin + self.loops[lidx].num_blocks;
                    self.loops[lidx].begin = begin;
                    self.loops[lidx].end = begin;
                    loop_begins[lidx] = Some(begin);
                }
            }

            // Place block
            let block_ref = block_rpo[loop_blocks[i].rpo_idx as usize];
            let block_idx = self.loops[loop_idx].end;
            self.loops[loop_idx].end = block_idx + 1;

            self.block_layout[block_idx as usize] = block_ref;
            self.block_loop_map[block_idx as usize] = loop_idx as u32;

            // Update adaptor's block info
            adaptor.set_block_idx(block_ref, block_idx as usize);
        }

        // Update block map to use layout indices
        self.block_map.clear();
        for (idx, &block) in self.block_layout.iter().enumerate() {
            self.block_map.insert(block, idx);
        }
    }
}
