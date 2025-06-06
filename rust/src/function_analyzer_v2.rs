//! Improved function analyzer that returns owned results.
//!
//! This version avoids lifetime issues by computing analysis results
//! and returning them as owned data rather than keeping references.

use crate::function_analysis_result::{FunctionAnalysisResult, PhiNodeInfo};
use crate::error::{CompileError, CompileResult};
use inkwell::basic_block::BasicBlock;
use inkwell::values::{FunctionValue, InstructionValue, InstructionOpcode};
use std::collections::HashMap;

/// Temporary analyzer that computes and returns results.
pub struct FunctionAnalyzerV2<'ctx> {
    blocks: Vec<BasicBlock<'ctx>>,
    block_map: HashMap<BasicBlock<'ctx>, usize>,
}

impl<'ctx> FunctionAnalyzerV2<'ctx> {
    /// Analyze a function and return owned results.
    pub fn analyze(function: FunctionValue<'ctx>) -> CompileResult<FunctionAnalysisResult> {
        let blocks = function.get_basic_blocks();
        let num_blocks = blocks.len();
        
        // Build block index map
        let mut block_map = HashMap::new();
        for (idx, &block) in blocks.iter().enumerate() {
            block_map.insert(block, idx);
        }
        
        let mut analyzer = Self {
            blocks: blocks.clone(),
            block_map,
        };
        
        analyzer.perform_analysis()
    }
    
    fn perform_analysis(&mut self) -> CompileResult<FunctionAnalysisResult> {
        let num_blocks = self.blocks.len();
        let mut result = FunctionAnalysisResult {
            num_blocks,
            block_layout: Vec::with_capacity(num_blocks),
            phi_count: 0,
            instruction_count: 0,
            block_phi_info: vec![(0, 0); num_blocks],
            phi_nodes: Vec::new(),
            block_has_branch: vec![false; num_blocks],
        };
        
        // First pass: collect basic information
        self.collect_block_info(&mut result)?;
        
        // Second pass: compute block layout (for now, just use natural order)
        // TODO: Implement proper reverse post-order traversal
        for i in 0..num_blocks {
            result.block_layout.push(i);
        }
        
        Ok(result)
    }
    
    fn collect_block_info(&self, result: &mut FunctionAnalysisResult) -> CompileResult<()> {
        let mut phi_start_idx = 0;
        
        for (block_idx, &block) in self.blocks.iter().enumerate() {
            let mut block_phi_count = 0;
            let block_phi_start = phi_start_idx;
            
            // Process instructions in the block
            for instruction in block.get_instructions() {
                result.instruction_count += 1;
                
                match instruction.get_opcode() {
                    InstructionOpcode::Phi => {
                        // Process PHI node
                        let phi_info = self.process_phi_node(instruction, block_idx)?;
                        result.phi_nodes.push(phi_info);
                        block_phi_count += 1;
                        result.phi_count += 1;
                    }
                    InstructionOpcode::Br => {
                        result.block_has_branch[block_idx] = true;
                    }
                    _ => {}
                }
            }
            
            // Store PHI info for this block
            result.block_phi_info[block_idx] = (block_phi_count, block_phi_start);
            phi_start_idx += block_phi_count;
        }
        
        Ok(())
    }
    
    fn process_phi_node(
        &self, 
        phi: InstructionValue<'ctx>, 
        block_idx: usize
    ) -> CompileResult<PhiNodeInfo> {
        // Get result value index
        use inkwell::values::AsValueRef;
        let result_idx = phi.as_value_ref() as usize % 1024;
        
        // Process incoming values
        let num_operands = phi.get_num_operands();
        let mut incoming = Vec::new();
        
        // PHI nodes have pairs of (value, block) operands
        for i in (0..num_operands).step_by(2) {
            if let (Some(value), Some(block)) = (
                phi.get_operand(i).and_then(|op| op.left()),
                phi.get_operand(i + 1).and_then(|op| op.right())
            ) {
                // Get value index
                let value_idx = value.as_value_ref() as usize % 1024;
                
                // Get predecessor block index
                let pred_idx = self.block_map.get(&block)
                    .copied()
                    .ok_or_else(|| CompileError::BlockLayout {
                        reason: "PHI references unknown block".to_string(),
                    })?;
                
                incoming.push((value_idx, pred_idx));
            }
        }
        
        Ok(PhiNodeInfo {
            block_idx,
            result_idx,
            incoming,
        })
    }
}