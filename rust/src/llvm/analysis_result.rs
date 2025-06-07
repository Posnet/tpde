// This module defines data structures for storing function analysis results that can be
// owned by the compiler, avoiding lifetime issues with the analyzer. FunctionAnalysisResult
// captures key analysis data: block count and layout order, PHI node information per block,
// instruction counts, and branch indicators. PhiNodeInfo stores details about individual
// PHI nodes including their block location, result value index, and incoming value/block
// pairs. This separation of analysis data from the analyzer allows the compiler to own
// and use the results throughout compilation without borrowing conflicts. The structures
// support efficient PHI node lookups by block and provide helper methods for querying
// PHI presence and retrieving nodes for specific blocks. This is essential for PHI node
// resolution during code generation.

//! Function analysis results that can be owned by the compiler.
//!
//! This separates the analysis data from the analyzer itself,
//! avoiding lifetime issues while preserving all the analysis information.

/// Results of function analysis that can be owned and passed around.
#[derive(Debug)]
pub struct FunctionAnalysisResult {
    /// Number of basic blocks.
    pub num_blocks: usize,

    /// Block layout order (optimized for control flow).
    /// Index in this vector is the layout position, value is the block index.
    pub block_layout: Vec<usize>,

    /// Number of PHI nodes in the function.
    pub phi_count: usize,

    /// Total number of instructions.
    pub instruction_count: usize,

    /// Information about PHI nodes per block.
    /// Indexed by block index, contains (phi_count, first_phi_index).
    pub block_phi_info: Vec<(usize, usize)>,

    /// PHI node details.
    pub phi_nodes: Vec<PhiNodeInfo>,

    /// Block has terminator that needs special handling.
    pub block_has_branch: Vec<bool>,
}

/// Information about a single PHI node.
#[derive(Debug, Clone)]
pub struct PhiNodeInfo {
    /// Block index containing this PHI.
    pub block_idx: usize,

    /// Result value index for this PHI.
    pub result_idx: usize,

    /// Incoming values: (value_idx, predecessor_block_idx).
    pub incoming: Vec<(usize, usize)>,
}

impl FunctionAnalysisResult {
    /// Create empty analysis results.
    pub fn empty() -> Self {
        Self {
            num_blocks: 0,
            block_layout: Vec::new(),
            phi_count: 0,
            instruction_count: 0,
            block_phi_info: Vec::new(),
            phi_nodes: Vec::new(),
            block_has_branch: Vec::new(),
        }
    }

    /// Get PHI nodes for a specific block.
    pub fn get_phi_nodes_for_block(&self, block_idx: usize) -> &[PhiNodeInfo] {
        if block_idx >= self.block_phi_info.len() {
            return &[];
        }

        let (count, start_idx) = self.block_phi_info[block_idx];
        if count == 0 {
            return &[];
        }

        &self.phi_nodes[start_idx..start_idx + count]
    }

    /// Check if a block has PHI nodes.
    pub fn block_has_phi_nodes(&self, block_idx: usize) -> bool {
        block_idx < self.block_phi_info.len() && self.block_phi_info[block_idx].0 > 0
    }
}
