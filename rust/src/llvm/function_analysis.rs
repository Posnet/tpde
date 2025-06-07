// This module implements function analysis for LLVM IR using arena allocation to minimize
// heap allocations and maximize performance. FunctionAnalyzer consumes itself to produce
// FunctionAnalysis results stored entirely in the compilation session arena. The analyzer
// performs three key passes: 1) Instruction analysis to identify PHI nodes and count
// instructions, 2) Successor extraction from terminator instructions (br, switch, invoke, etc.)
// to build the control flow graph, 3) Block layout computation (currently natural order,
// RPO planned). It tracks PHI node information including incoming values and predecessor
// blocks, block successor/predecessor relationships for control flow, and provides efficient
// queries for PHI nodes per block and successor lookup. The arena-based design ensures all
// analysis data has unified lifetime tied to the compilation session, avoiding complex
// lifetime management while maintaining performance.

//! Function analyzer that uses arena allocation and consumes itself to produce results.
//!
//! This design avoids copies and heap allocations by using arena allocation
//! throughout and moving ownership when producing results.

use crate::core::error::{CompileError, CompileResult};
use crate::core::session::CompilationSession;
use bumpalo::collections::Vec as BumpVec;
use inkwell::basic_block::BasicBlock;
use inkwell::values::{FunctionValue, InstructionOpcode, InstructionValue};

/// Analysis results that own arena-allocated data.
pub struct FunctionAnalysis<'arena> {
    /// Number of blocks analyzed.
    pub num_blocks: usize,

    /// Block layout order (reverse post-order).
    pub block_layout: &'arena [usize],

    /// PHI node information per block: (count, start_index).
    pub block_phi_info: &'arena [(usize, usize)],

    /// All PHI nodes in the function.
    pub phi_nodes: &'arena [PhiNode],

    /// Incoming values for PHI nodes.
    pub phi_incoming: &'arena [PhiIncoming],

    /// Total PHI count.
    pub phi_count: usize,

    /// Total instruction count.
    pub instruction_count: usize,

    /// Block successor information.
    pub block_successors: &'arena [BlockSuccessors],

    /// Successor indices array.
    pub successor_indices: &'arena [usize],

    /// Block predecessor count (for identifying join blocks).
    pub block_predecessor_count: &'arena [usize],
}

/// Block successor information.
#[derive(Debug, Clone, Copy)]
pub struct BlockSuccessors {
    /// Index of first successor in successors array.
    pub start_idx: usize,

    /// Number of successors.
    pub count: usize,

    /// Whether this block has conditional control flow.
    pub is_conditional: bool,
}

/// Information about a PHI node.
#[derive(Debug, Clone, Copy)]
pub struct PhiNode {
    /// Block containing this PHI.
    pub block_idx: usize,

    /// Result value index.
    pub result_idx: usize,

    /// Start index in incoming_values array.
    pub incoming_start: usize,

    /// Number of incoming values.
    pub incoming_count: usize,
}

/// Incoming value for a PHI node.
#[derive(Debug, Clone, Copy)]
pub struct PhiIncoming {
    /// Value index.
    pub value_idx: usize,

    /// Predecessor block index.
    pub pred_block_idx: usize,
}

/// Builder that consumes itself to produce analysis results.
pub struct FunctionAnalyzer<'ctx, 'arena> {
    _session: &'arena CompilationSession<'arena>,
    blocks: Vec<BasicBlock<'ctx>>,

    // Working data - will be moved to results
    block_layout: BumpVec<'arena, usize>,
    block_phi_info: BumpVec<'arena, (usize, usize)>,
    phi_nodes: BumpVec<'arena, PhiNode>,
    phi_incoming: BumpVec<'arena, PhiIncoming>,
    block_successors: BumpVec<'arena, BlockSuccessors>,
    successor_indices: BumpVec<'arena, usize>,
    block_predecessor_count: BumpVec<'arena, usize>,

    // Counters
    phi_count: usize,
    phi_incoming_count: usize,
    instruction_count: usize,
    successor_count: usize,
}

impl<'ctx, 'arena> FunctionAnalyzer<'ctx, 'arena> {
    const MAX_BLOCKS: usize = 256;
    const MAX_PHI_NODES: usize = 128;
    const MAX_PHI_INCOMING: usize = 512;

    /// Create a new analyzer that borrows the arena.
    pub fn new(
        session: &'arena CompilationSession<'arena>,
        function: FunctionValue<'ctx>,
    ) -> CompileResult<Self> {
        let blocks = function.get_basic_blocks();
        let num_blocks = blocks.len();

        if num_blocks > Self::MAX_BLOCKS {
            return Err(CompileError::BlockLayout {
                reason: format!("Too many blocks: {}", num_blocks),
            });
        }

        Ok(Self {
            _session: session,
            blocks,
            block_layout: {
                let mut v = BumpVec::with_capacity_in(num_blocks, session.arena());
                v.resize(num_blocks, 0);
                v
            },
            block_phi_info: {
                let mut v = BumpVec::with_capacity_in(num_blocks, session.arena());
                v.resize(num_blocks, (0, 0));
                v
            },
            phi_nodes: BumpVec::with_capacity_in(Self::MAX_PHI_NODES, session.arena()),
            phi_incoming: BumpVec::with_capacity_in(Self::MAX_PHI_INCOMING, session.arena()),
            block_successors: {
                let mut v = BumpVec::with_capacity_in(num_blocks, session.arena());
                v.resize(
                    num_blocks,
                    BlockSuccessors {
                        start_idx: 0,
                        count: 0,
                        is_conditional: false,
                    },
                );
                v
            },
            successor_indices: BumpVec::with_capacity_in(num_blocks * 2, session.arena()),
            block_predecessor_count: {
                let mut v = BumpVec::with_capacity_in(num_blocks, session.arena());
                v.resize(num_blocks, 0);
                v
            },
            phi_count: 0,
            phi_incoming_count: 0,
            instruction_count: 0,
            successor_count: 0,
        })
    }

    /// Perform analysis and consume self to produce results.
    pub fn analyze(mut self) -> CompileResult<FunctionAnalysis<'arena>> {
        // First pass: analyze instructions, PHI nodes, and successors
        self.analyze_instructions()?;

        // Second pass: extract block successors
        self.extract_block_successors()?;

        // Third pass: compute block layout
        self.compute_block_layout()?;

        // Convert bump-allocated working data into slices
        let num_blocks = self.blocks.len();
        let block_layout = self.block_layout.into_bump_slice();
        let block_phi_info = self.block_phi_info.into_bump_slice();
        let phi_nodes = self.phi_nodes.into_bump_slice();
        let phi_incoming = self.phi_incoming.into_bump_slice();
        let block_successors = self.block_successors.into_bump_slice();
        let successor_indices = self.successor_indices.into_bump_slice();
        let block_predecessor_count = self.block_predecessor_count.into_bump_slice();

        Ok(FunctionAnalysis {
            num_blocks,
            block_layout,
            block_phi_info,
            phi_nodes,
            phi_incoming,
            phi_count: self.phi_count,
            instruction_count: self.instruction_count,
            block_successors,
            successor_indices,
            block_predecessor_count,
        })
    }

    fn analyze_instructions(&mut self) -> CompileResult<()> {
        for block_idx in 0..self.blocks.len() {
            let block = self.blocks[block_idx];
            let phi_start = self.phi_count;
            let mut block_phi_count = 0;

            for instruction in block.get_instructions() {
                self.instruction_count += 1;

                if instruction.get_opcode() == InstructionOpcode::Phi {
                    self.process_phi_node(instruction, block_idx)?;
                    block_phi_count += 1;
                }
            }

            self.block_phi_info[block_idx] = (block_phi_count, phi_start);
        }

        Ok(())
    }

    fn process_phi_node(
        &mut self,
        phi: InstructionValue<'ctx>,
        block_idx: usize,
    ) -> CompileResult<()> {
        if self.phi_count >= Self::MAX_PHI_NODES {
            return Err(CompileError::BlockLayout {
                reason: "Too many PHI nodes".to_string(),
            });
        }

        // Get result index
        use inkwell::values::AsValueRef;
        let result_idx = phi.as_value_ref() as usize % 1024;

        let incoming_start = self.phi_incoming_count;
        let mut incoming_count = 0;

        // Process incoming values using PhiValue methods
        use inkwell::values::PhiValue;
        if let Ok(phi_value) = PhiValue::try_from(phi) {
            let num_incoming = phi_value.count_incoming();

            for i in 0..num_incoming {
                if let Some((value, block)) = phi_value.get_incoming(i) {
                    if self.phi_incoming_count >= Self::MAX_PHI_INCOMING {
                        return Err(CompileError::BlockLayout {
                            reason: "Too many PHI incoming values".to_string(),
                        });
                    }

                    let value_idx = value.as_value_ref() as usize % 1024;
                    let pred_idx = self.find_block_index(block)?;

                    self.phi_incoming.push(PhiIncoming {
                        value_idx,
                        pred_block_idx: pred_idx,
                    });
                    self.phi_incoming_count += 1;
                    incoming_count += 1;
                }
            }
        } else {
            // Fallback to operand-based extraction
            let num_operands = phi.get_num_operands();
            for i in (0..num_operands).step_by(2) {
                if let (Some(value), Some(block)) = (
                    phi.get_operand(i).and_then(|op| op.left()),
                    phi.get_operand(i + 1).and_then(|op| op.right()),
                ) {
                    if self.phi_incoming_count >= Self::MAX_PHI_INCOMING {
                        return Err(CompileError::BlockLayout {
                            reason: "Too many PHI incoming values".to_string(),
                        });
                    }

                    let value_idx = value.as_value_ref() as usize % 1024;
                    let pred_idx = self.find_block_index(block)?;

                    self.phi_incoming.push(PhiIncoming {
                        value_idx,
                        pred_block_idx: pred_idx,
                    });
                    self.phi_incoming_count += 1;
                    incoming_count += 1;
                }
            }
        }

        self.phi_nodes.push(PhiNode {
            block_idx,
            result_idx,
            incoming_start,
            incoming_count,
        });
        self.phi_count += 1;

        Ok(())
    }

    fn find_block_index(&self, block: BasicBlock<'ctx>) -> CompileResult<usize> {
        self.blocks
            .iter()
            .position(|&b| b == block)
            .ok_or_else(|| CompileError::BlockLayout {
                reason: "PHI references unknown block".to_string(),
            })
    }

    fn extract_block_successors(&mut self) -> CompileResult<()> {
        for block_idx in 0..self.blocks.len() {
            let block = self.blocks[block_idx];

            // Get terminator instruction
            if let Some(terminator) = block.get_terminator() {
                let start_idx = self.successor_count;
                let mut count = 0;
                let mut is_conditional = false;

                match terminator.get_opcode() {
                    InstructionOpcode::Br => {
                        let num_operands = terminator.get_num_operands();

                        if num_operands == 1 {
                            // Unconditional branch
                            if let Some(target) =
                                terminator.get_operand(0).and_then(|op| op.right())
                            {
                                let target_idx = self.find_block_index(target)?;
                                self.successor_indices.push(target_idx);
                                self.block_predecessor_count[target_idx] += 1;
                                count = 1;
                            }
                        } else if num_operands == 3 {
                            // Conditional branch
                            is_conditional = true;

                            // True target (operand 2 in LLVM)
                            if let Some(true_target) =
                                terminator.get_operand(2).and_then(|op| op.right())
                            {
                                let target_idx = self.find_block_index(true_target)?;
                                self.successor_indices.push(target_idx);
                                self.block_predecessor_count[target_idx] += 1;
                                count += 1;
                            }

                            // False target (operand 1 in LLVM)
                            if let Some(false_target) =
                                terminator.get_operand(1).and_then(|op| op.right())
                            {
                                let target_idx = self.find_block_index(false_target)?;
                                self.successor_indices.push(target_idx);
                                self.block_predecessor_count[target_idx] += 1;
                                count += 1;
                            }
                        }
                    }
                    InstructionOpcode::Switch => {
                        // Switch instruction has multiple targets
                        is_conditional = true;

                        // Default target is operand 1
                        if let Some(default_target) =
                            terminator.get_operand(1).and_then(|op| op.right())
                        {
                            let target_idx = self.find_block_index(default_target)?;
                            self.successor_indices.push(target_idx);
                            self.block_predecessor_count[target_idx] += 1;
                            count += 1;
                        }

                        // Case targets are at operands 3, 5, 7, ...
                        for i in (3..terminator.get_num_operands()).step_by(2) {
                            if let Some(case_target) =
                                terminator.get_operand(i).and_then(|op| op.right())
                            {
                                let target_idx = self.find_block_index(case_target)?;
                                self.successor_indices.push(target_idx);
                                self.block_predecessor_count[target_idx] += 1;
                                count += 1;
                            }
                        }
                    }
                    InstructionOpcode::IndirectBr => {
                        // Indirect branch - all operands after the address are potential targets
                        is_conditional = true;

                        for i in 1..terminator.get_num_operands() {
                            if let Some(target) =
                                terminator.get_operand(i).and_then(|op| op.right())
                            {
                                let target_idx = self.find_block_index(target)?;
                                self.successor_indices.push(target_idx);
                                self.block_predecessor_count[target_idx] += 1;
                                count += 1;
                            }
                        }
                    }
                    InstructionOpcode::Invoke => {
                        // Invoke has normal and exception destinations
                        is_conditional = true;

                        // Normal destination
                        if let Some(normal) = terminator.get_operand(1).and_then(|op| op.right()) {
                            let target_idx = self.find_block_index(normal)?;
                            self.successor_indices.push(target_idx);
                            self.block_predecessor_count[target_idx] += 1;
                            count += 1;
                        }

                        // Exception destination
                        if let Some(exception) = terminator.get_operand(2).and_then(|op| op.right())
                        {
                            let target_idx = self.find_block_index(exception)?;
                            self.successor_indices.push(target_idx);
                            self.block_predecessor_count[target_idx] += 1;
                            count += 1;
                        }
                    }
                    InstructionOpcode::Return | InstructionOpcode::Unreachable => {
                        // No successors
                        count = 0;
                    }
                    _ => {
                        // Unknown terminator
                        return Err(CompileError::BlockLayout {
                            reason: format!(
                                "Unknown terminator opcode: {:?}",
                                terminator.get_opcode()
                            ),
                        });
                    }
                }

                self.block_successors[block_idx] = BlockSuccessors {
                    start_idx,
                    count,
                    is_conditional,
                };
                self.successor_count += count;
            }
        }

        Ok(())
    }

    fn compute_block_layout(&mut self) -> CompileResult<()> {
        // For now, use natural order
        // TODO: Implement reverse post-order
        for i in 0..self.blocks.len() {
            self.block_layout[i] = i;
        }
        Ok(())
    }
}

impl<'arena> FunctionAnalysis<'arena> {
    /// Get PHI nodes for a specific block.
    pub fn get_block_phi_nodes(&self, block_idx: usize) -> &[PhiNode] {
        if block_idx >= self.block_phi_info.len() {
            return &[];
        }

        let (count, start) = self.block_phi_info[block_idx];
        if count == 0 || start >= self.phi_nodes.len() {
            return &[];
        }

        let end = (start + count).min(self.phi_nodes.len());
        &self.phi_nodes[start..end]
    }

    /// Get incoming values for a PHI node.
    pub fn get_phi_incoming(&self, phi: &PhiNode) -> &[PhiIncoming] {
        &self.phi_incoming[phi.incoming_start..phi.incoming_start + phi.incoming_count]
    }

    /// Get successor blocks for a specific block.
    pub fn get_block_successors(&self, block_idx: usize) -> &[usize] {
        if block_idx >= self.block_successors.len() {
            return &[];
        }

        let info = &self.block_successors[block_idx];
        if info.count == 0 {
            return &[];
        }

        let end_idx = (info.start_idx + info.count).min(self.successor_indices.len());
        &self.successor_indices[info.start_idx..end_idx]
    }

    /// Check if a block has multiple predecessors (is a join block).
    pub fn is_join_block(&self, block_idx: usize) -> bool {
        block_idx < self.block_predecessor_count.len()
            && self.block_predecessor_count[block_idx] > 1
    }

    /// Check if a block has conditional control flow.
    pub fn has_conditional_branch(&self, block_idx: usize) -> bool {
        block_idx < self.block_successors.len() && self.block_successors[block_idx].is_conditional
    }
}
