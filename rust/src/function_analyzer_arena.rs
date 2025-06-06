//! Function analyzer that uses arena allocation and consumes itself to produce results.
//!
//! This design avoids copies and heap allocations by using arena allocation
//! throughout and moving ownership when producing results.

use crate::compilation_session::CompilationSession;
use crate::error::{CompileError, CompileResult};
use inkwell::basic_block::BasicBlock;
use inkwell::values::{FunctionValue, InstructionValue, InstructionOpcode};

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
    session: &'arena CompilationSession<'arena>,
    blocks: Vec<BasicBlock<'ctx>>,
    
    // Working data - will be moved to results
    block_layout: Vec<usize>,
    block_phi_info: Vec<(usize, usize)>,
    phi_nodes: Vec<PhiNode>,
    phi_incoming: Vec<PhiIncoming>,
    
    // Counters
    phi_count: usize,
    phi_incoming_count: usize,
    instruction_count: usize,
}

impl<'ctx, 'arena> FunctionAnalyzer<'ctx, 'arena> {
    const MAX_BLOCKS: usize = 256;
    const MAX_PHI_NODES: usize = 128;
    const MAX_PHI_INCOMING: usize = 512;
    
    /// Create a new analyzer that borrows the arena.
    pub fn new(
        session: &'arena CompilationSession<'arena>,
        function: FunctionValue<'ctx>
    ) -> CompileResult<Self> {
        let blocks = function.get_basic_blocks();
        let num_blocks = blocks.len();
        
        if num_blocks > Self::MAX_BLOCKS {
            return Err(CompileError::BlockLayout {
                reason: format!("Too many blocks: {}", num_blocks),
            });
        }
        
        Ok(Self {
            session,
            blocks,
            block_layout: vec![0; num_blocks],
            block_phi_info: vec![(0, 0); num_blocks],
            phi_nodes: Vec::with_capacity(Self::MAX_PHI_NODES),
            phi_incoming: Vec::with_capacity(Self::MAX_PHI_INCOMING),
            phi_count: 0,
            phi_incoming_count: 0,
            instruction_count: 0,
        })
    }
    
    /// Perform analysis and consume self to produce results.
    pub fn analyze(mut self) -> CompileResult<FunctionAnalysis<'arena>> {
        // First pass: analyze instructions and PHI nodes
        self.analyze_instructions()?;
        
        // Second pass: compute block layout
        self.compute_block_layout()?;
        
        // Allocate results in arena
        let num_blocks = self.blocks.len();
        let block_layout = self.session.alloc_slice(&self.block_layout[..num_blocks]);
        let block_phi_info = self.session.alloc_slice(&self.block_phi_info[..num_blocks]);
        let phi_nodes = self.session.alloc_slice(&self.phi_nodes[..self.phi_count]);
        let phi_incoming = self.session.alloc_slice(&self.phi_incoming[..self.phi_incoming_count]);
        
        Ok(FunctionAnalysis {
            num_blocks,
            block_layout,
            block_phi_info,
            phi_nodes,
            phi_incoming,
            phi_count: self.phi_count,
            instruction_count: self.instruction_count,
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
        block_idx: usize
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
        
        // Process incoming values
        let num_operands = phi.get_num_operands();
        for i in (0..num_operands).step_by(2) {
            if let (Some(value), Some(block)) = (
                phi.get_operand(i).and_then(|op| op.left()),
                phi.get_operand(i + 1).and_then(|op| op.right())
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
        self.blocks.iter().position(|&b| b == block)
            .ok_or_else(|| CompileError::BlockLayout {
                reason: "PHI references unknown block".to_string(),
            })
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
}

