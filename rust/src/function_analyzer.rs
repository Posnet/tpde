//! Function analyzer for block layout and liveness analysis.
//!
//! This module implements the analysis phase that the C++ version uses
//! to optimize block layout and prepare for PHI node resolution.
//!
//! All allocations are done through the arena to avoid heap allocations.

use crate::compilation_session::CompilationSession;
use crate::error::{CompileError, CompileResult};
use inkwell::basic_block::BasicBlock;
use inkwell::values::{FunctionValue, InstructionValue, InstructionOpcode};

/// Block analysis information stored in arena.
#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    /// Index in layout order.
    pub layout_index: usize,
    
    /// Number of predecessors.
    pub predecessor_count: usize,
    
    /// Number of successors.
    pub successor_count: usize,
    
    /// Number of PHI nodes in this block.
    pub phi_node_count: usize,
    
    /// Whether this block is a loop header.
    pub is_loop_header: bool,
    
    /// Loop depth (0 for non-loop blocks).
    pub loop_depth: usize,
}

/// PHI node information stored in arena.
#[derive(Debug, Clone, Copy)]
pub struct PhiInfo {
    /// Block index containing this PHI.
    pub block_idx: usize,
    
    /// Instruction index of the PHI node.
    pub inst_idx: usize,
    
    /// Number of incoming values.
    pub incoming_count: usize,
    
    /// Result value index.
    pub result_idx: usize,
}

/// Incoming value for a PHI node.
#[derive(Debug, Clone, Copy)]
pub struct PhiIncoming {
    /// PHI node index this belongs to.
    pub phi_idx: usize,
    
    /// Value index.
    pub value_idx: usize,
    
    /// Predecessor block index.
    pub pred_block_idx: usize,
}

/// Edge in the control flow graph.
#[derive(Debug, Clone, Copy)]
pub struct BlockEdge {
    /// Source block index.
    pub from: usize,
    
    /// Target block index.
    pub to: usize,
}

/// Function analyzer that performs block layout and analysis.
pub struct FunctionAnalyzer<'ctx, 'arena> {
    /// Compilation session for arena allocation.
    session: &'arena CompilationSession<'arena>,
    
    /// Basic blocks from the function.
    pub blocks: &'arena [BasicBlock<'ctx>],
    
    /// Block information array (indexed by block index).
    block_info: &'arena mut [BlockInfo],
    
    /// All edges in the CFG.
    edges: &'arena mut [BlockEdge],
    
    /// Number of edges actually used.
    edge_count: usize,
    
    /// PHI node information.
    phi_nodes: &'arena mut [PhiInfo],
    
    /// Number of PHI nodes.
    pub phi_count: usize,
    
    /// PHI incoming values.
    phi_incoming: &'arena mut [PhiIncoming],
    
    /// Number of PHI incoming values.
    phi_incoming_count: usize,
    
    /// Block layout order (optimized for control flow).
    pub block_layout: &'arena mut [usize],
    
    /// Total number of instructions.
    pub instruction_count: usize,
}

impl<'ctx, 'arena> FunctionAnalyzer<'ctx, 'arena> {
    /// Maximum number of blocks we support.
    const MAX_BLOCKS: usize = 256;
    
    /// Maximum number of edges (generous allocation).
    const MAX_EDGES: usize = 512;
    
    /// Maximum number of PHI nodes.
    const MAX_PHI_NODES: usize = 128;
    
    /// Maximum number of PHI incoming values.
    const MAX_PHI_INCOMING: usize = 512;
    
    /// Analyze a function and compute optimal block layout.
    pub fn analyze(
        session: &'arena CompilationSession<'arena>,
        function: FunctionValue<'ctx>
    ) -> CompileResult<Self> {
        let blocks = function.get_basic_blocks();
        let num_blocks = blocks.len();
        
        if num_blocks > Self::MAX_BLOCKS {
            return Err(CompileError::BlockLayout {
                reason: format!("Too many blocks: {} > {}", num_blocks, Self::MAX_BLOCKS),
            });
        }
        
        // Allocate all arrays in the arena
        let blocks = session.alloc_slice(&blocks);
        let block_info = session.alloc_slice(&vec![BlockInfo {
            layout_index: 0,
            predecessor_count: 0,
            successor_count: 0,
            phi_node_count: 0,
            is_loop_header: false,
            loop_depth: 0,
        }; num_blocks]);
        
        // SAFETY: We need mutable slices, but we know we're the only ones accessing them
        let block_info = unsafe {
            std::slice::from_raw_parts_mut(block_info.as_ptr() as *mut BlockInfo, block_info.len())
        };
        
        let edges = session.alloc_slice(&vec![BlockEdge { from: 0, to: 0 }; Self::MAX_EDGES]);
        let edges = unsafe {
            std::slice::from_raw_parts_mut(edges.as_ptr() as *mut BlockEdge, edges.len())
        };
        
        let phi_nodes = session.alloc_slice(&vec![PhiInfo {
            block_idx: 0,
            inst_idx: 0,
            incoming_count: 0,
            result_idx: 0,
        }; Self::MAX_PHI_NODES]);
        let phi_nodes = unsafe {
            std::slice::from_raw_parts_mut(phi_nodes.as_ptr() as *mut PhiInfo, phi_nodes.len())
        };
        
        let phi_incoming = session.alloc_slice(&vec![PhiIncoming {
            phi_idx: 0,
            value_idx: 0,
            pred_block_idx: 0,
        }; Self::MAX_PHI_INCOMING]);
        let phi_incoming = unsafe {
            std::slice::from_raw_parts_mut(phi_incoming.as_ptr() as *mut PhiIncoming, phi_incoming.len())
        };
        
        let block_layout = session.alloc_slice(&vec![0; num_blocks]);
        let block_layout = unsafe {
            std::slice::from_raw_parts_mut(block_layout.as_ptr() as *mut usize, block_layout.len())
        };
        
        let mut analyzer = Self {
            session,
            blocks,
            block_info,
            edges,
            edge_count: 0,
            phi_nodes,
            phi_count: 0,
            phi_incoming,
            phi_incoming_count: 0,
            block_layout,
            instruction_count: 0,
        };
        
        analyzer.analyze_control_flow(function)?;
        analyzer.compute_layout()?;
        analyzer.detect_loops()?;
        
        Ok(analyzer)
    }
    
    /// Analyze control flow and build predecessor/successor relationships.
    fn analyze_control_flow(&mut self, _function: FunctionValue<'ctx>) -> CompileResult<()> {
        // First pass: analyze terminators and build edges
        for (block_idx, block) in self.blocks.iter().enumerate() {
            // Find terminator instruction
            if let Some(terminator) = block.get_terminator() {
                self.extract_successors(block_idx, terminator)?;
            }
            
            // Count instructions and find PHI nodes
            let mut inst_idx = self.instruction_count;
            for instruction in block.get_instructions() {
                if instruction.get_opcode() == InstructionOpcode::Phi {
                    self.analyze_phi_node(block_idx, inst_idx, instruction)?;
                }
                inst_idx += 1;
            }
            self.instruction_count = inst_idx;
        }
        
        // Second pass: count predecessors from edges
        for i in 0..self.edge_count {
            let edge = self.edges[i];
            self.block_info[edge.to].predecessor_count += 1;
        }
        
        Ok(())
    }
    
    /// Extract successor blocks from a terminator instruction.
    fn extract_successors(
        &mut self,
        block_idx: usize,
        terminator: InstructionValue<'ctx>
    ) -> CompileResult<()> {
        match terminator.get_opcode() {
            InstructionOpcode::Br => {
                // Branch can be conditional or unconditional
                let num_operands = terminator.get_num_operands();
                if num_operands == 1 {
                    // Unconditional branch
                    if let Some(target) = terminator.get_operand(0).and_then(|op| op.right()) {
                        if let Some(target_idx) = self.find_block_index(target) {
                            self.add_edge(block_idx, target_idx)?;
                        }
                    }
                } else if num_operands == 3 {
                    // Conditional branch
                    if let Some(true_target) = terminator.get_operand(2).and_then(|op| op.right()) {
                        if let Some(target_idx) = self.find_block_index(true_target) {
                            self.add_edge(block_idx, target_idx)?;
                        }
                    }
                    if let Some(false_target) = terminator.get_operand(1).and_then(|op| op.right()) {
                        if let Some(target_idx) = self.find_block_index(false_target) {
                            self.add_edge(block_idx, target_idx)?;
                        }
                    }
                }
            }
            InstructionOpcode::Switch => {
                // Default target is operand 1
                if let Some(default) = terminator.get_operand(1).and_then(|op| op.right()) {
                    if let Some(target_idx) = self.find_block_index(default) {
                        self.add_edge(block_idx, target_idx)?;
                    }
                }
                // Case targets are in pairs starting at operand 2
                let num_operands = terminator.get_num_operands();
                for i in (2..num_operands).step_by(2) {
                    if let Some(case_target) = terminator.get_operand(i + 1).and_then(|op| op.right()) {
                        if let Some(target_idx) = self.find_block_index(case_target) {
                            // Check if edge already exists to avoid duplicates
                            let mut exists = false;
                            for j in 0..self.edge_count {
                                if self.edges[j].from == block_idx && self.edges[j].to == target_idx {
                                    exists = true;
                                    break;
                                }
                            }
                            if !exists {
                                self.add_edge(block_idx, target_idx)?;
                            }
                        }
                    }
                }
            }
            InstructionOpcode::Return | InstructionOpcode::Unreachable => {
                // No successors
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Find block index by BasicBlock reference.
    fn find_block_index(&self, block: BasicBlock<'ctx>) -> Option<usize> {
        self.blocks.iter().position(|&b| b == block)
    }
    
    /// Add an edge to the CFG.
    fn add_edge(&mut self, from: usize, to: usize) -> CompileResult<()> {
        if self.edge_count >= Self::MAX_EDGES {
            return Err(CompileError::BlockLayout {
                reason: "Too many edges in CFG".to_string(),
            });
        }
        
        self.edges[self.edge_count] = BlockEdge { from, to };
        self.edge_count += 1;
        self.block_info[from].successor_count += 1;
        
        Ok(())
    }
    
    /// Analyze a PHI node instruction.
    fn analyze_phi_node(
        &mut self, 
        block_idx: usize, 
        inst_idx: usize,
        phi: InstructionValue<'ctx>
    ) -> CompileResult<()> {
        if self.phi_count >= Self::MAX_PHI_NODES {
            return Err(CompileError::BlockLayout {
                reason: "Too many PHI nodes".to_string(),
            });
        }
        
        let phi_idx = self.phi_count;
        let _incoming_start = self.phi_incoming_count;
        
        // PHI nodes have pairs of (value, block) operands
        let num_operands = phi.get_num_operands();
        let mut incoming_count = 0;
        
        for i in (0..num_operands).step_by(2) {
            if let (Some(value), Some(block)) = (
                phi.get_operand(i).and_then(|op| op.left()),
                phi.get_operand(i + 1).and_then(|op| op.right())
            ) {
                if let Some(pred_idx) = self.find_block_index(block) {
                    if self.phi_incoming_count >= Self::MAX_PHI_INCOMING {
                        return Err(CompileError::BlockLayout {
                            reason: "Too many PHI incoming values".to_string(),
                        });
                    }
                    
                    // Use simplified value indexing for now
                    use inkwell::values::AsValueRef;
                    let value_idx = value.as_value_ref() as usize % 1024;
                    
                    self.phi_incoming[self.phi_incoming_count] = PhiIncoming {
                        phi_idx,
                        value_idx,
                        pred_block_idx: pred_idx,
                    };
                    self.phi_incoming_count += 1;
                    incoming_count += 1;
                }
            }
        }
        
        // Result value index
        use inkwell::values::AsValueRef;
        let result_idx = phi.as_value_ref() as usize % 1024;
        
        self.phi_nodes[phi_idx] = PhiInfo {
            block_idx,
            inst_idx,
            incoming_count,
            result_idx,
        };
        
        self.block_info[block_idx].phi_node_count += 1;
        self.phi_count += 1;
        
        Ok(())
    }
    
    /// Compute optimized block layout using reverse post-order.
    fn compute_layout(&mut self) -> CompileResult<()> {
        if self.blocks.is_empty() {
            return Ok(());
        }
        
        let num_blocks = self.blocks.len();
        
        // Use arena-allocated arrays for DFS
        let visited = self.session.alloc_slice(&vec![false; num_blocks]);
        let post_order = self.session.alloc_slice(&vec![0usize; num_blocks]);
        
        // SAFETY: We need mutable access
        let visited = unsafe {
            std::slice::from_raw_parts_mut(visited.as_ptr() as *mut bool, visited.len())
        };
        let post_order = unsafe {
            std::slice::from_raw_parts_mut(post_order.as_ptr() as *mut usize, post_order.len())
        };
        
        let mut post_order_count = 0;
        self.dfs_post_order(0, visited, post_order, &mut post_order_count);
        
        // Reverse to get reverse post-order
        for i in 0..post_order_count / 2 {
            post_order.swap(i, post_order_count - 1 - i);
        }
        
        // Copy to block_layout
        self.block_layout[..post_order_count].copy_from_slice(&post_order[..post_order_count]);
        
        // Update layout indices
        for (new_idx, &block_idx) in self.block_layout[..num_blocks].iter().enumerate() {
            self.block_info[block_idx].layout_index = new_idx;
        }
        
        Ok(())
    }
    
    /// DFS traversal for post-order.
    fn dfs_post_order(
        &self,
        block_idx: usize,
        visited: &mut [bool],
        post_order: &mut [usize],
        post_order_count: &mut usize
    ) {
        if visited[block_idx] {
            return;
        }
        
        visited[block_idx] = true;
        
        // Visit successors
        for i in 0..self.edge_count {
            if self.edges[i].from == block_idx {
                self.dfs_post_order(self.edges[i].to, visited, post_order, post_order_count);
            }
        }
        
        post_order[*post_order_count] = block_idx;
        *post_order_count += 1;
    }
    
    /// Detect loops using back edge detection.
    fn detect_loops(&mut self) -> CompileResult<()> {
        let num_blocks = self.blocks.len();
        
        // Use arena-allocated arrays
        let visited = self.session.alloc_slice(&vec![false; num_blocks]);
        let on_stack = self.session.alloc_slice(&vec![false; num_blocks]);
        
        // SAFETY: We need mutable access
        let visited = unsafe {
            std::slice::from_raw_parts_mut(visited.as_ptr() as *mut bool, visited.len())
        };
        let on_stack = unsafe {
            std::slice::from_raw_parts_mut(on_stack.as_ptr() as *mut bool, on_stack.len())
        };
        
        for block_idx in 0..num_blocks {
            if !visited[block_idx] {
                self.detect_loops_dfs(block_idx, visited, on_stack)?;
            }
        }
        
        Ok(())
    }
    
    /// DFS for loop detection.
    fn detect_loops_dfs(
        &mut self,
        block_idx: usize,
        visited: &mut [bool],
        on_stack: &mut [bool]
    ) -> CompileResult<()> {
        visited[block_idx] = true;
        on_stack[block_idx] = true;
        
        // Check all successors
        for i in 0..self.edge_count {
            if self.edges[i].from == block_idx {
                let succ_idx = self.edges[i].to;
                
                if !visited[succ_idx] {
                    self.detect_loops_dfs(succ_idx, visited, on_stack)?;
                } else if on_stack[succ_idx] {
                    // Back edge found: succ_idx is a loop header
                    self.block_info[succ_idx].is_loop_header = true;
                    // Simple loop depth marking
                    self.block_info[succ_idx].loop_depth += 1;
                }
            }
        }
        
        on_stack[block_idx] = false;
        Ok(())
    }
    
    /// Get edges for a specific block.
    pub fn get_successors(&self, block_idx: usize) -> &[BlockEdge] {
        let mut start = 0;
        let mut count = 0;
        
        for i in 0..self.edge_count {
            if self.edges[i].from == block_idx {
                if count == 0 {
                    start = i;
                }
                count += 1;
            } else if count > 0 {
                break;
            }
        }
        
        &self.edges[start..start + count]
    }
    
    /// Get PHI nodes for a specific block.
    pub fn get_phi_nodes(&self, block_idx: usize) -> &[PhiInfo] {
        let mut start = 0;
        let mut count = 0;
        
        for i in 0..self.phi_count {
            if self.phi_nodes[i].block_idx == block_idx {
                if count == 0 {
                    start = i;
                }
                count += 1;
            } else if count > 0 {
                break;
            }
        }
        
        &self.phi_nodes[start..start + count]
    }
    
    /// Get incoming values for a PHI node.
    pub fn get_phi_incoming(&self, phi_idx: usize) -> &[PhiIncoming] {
        let mut start = 0;
        let mut count = 0;
        
        for i in 0..self.phi_incoming_count {
            if self.phi_incoming[i].phi_idx == phi_idx {
                if count == 0 {
                    start = i;
                }
                count += 1;
            } else if count > 0 {
                break;
            }
        }
        
        &self.phi_incoming[start..start + count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;
    use inkwell::IntPredicate;
    use bumpalo::Bump;
    
    #[test]
    fn test_simple_function_analysis() {
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        
        // Create a simple function with branches
        let fn_type = i32_type.fn_type(&[i32_type.into()], false);
        let function = module.add_function("test_func", fn_type, None);
        
        let entry = context.append_basic_block(function, "entry");
        let then_block = context.append_basic_block(function, "then");
        let else_block = context.append_basic_block(function, "else");
        
        let builder = context.create_builder();
        
        // Entry block
        builder.position_at_end(entry);
        let param = function.get_nth_param(0).unwrap().into_int_value();
        let zero = i32_type.const_zero();
        let cond = builder.build_int_compare(IntPredicate::EQ, param, zero, "cond").unwrap();
        builder.build_conditional_branch(cond, then_block, else_block).unwrap();
        
        // Then block
        builder.position_at_end(then_block);
        let one = i32_type.const_int(1, false);
        builder.build_return(Some(&one)).unwrap();
        
        // Else block
        builder.position_at_end(else_block);
        let two = i32_type.const_int(2, false);
        builder.build_return(Some(&two)).unwrap();
        
        // Create arena and session
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);
        
        // Analyze the function
        let analyzer = FunctionAnalyzer::analyze(&session, function).unwrap();
        
        // Verify analysis results
        assert_eq!(analyzer.blocks.len(), 3);
        assert_eq!(analyzer.edge_count, 2); // entry->then, entry->else
        
        // Entry block should have 2 successors
        let entry_info = &analyzer.block_info[0];
        assert_eq!(entry_info.successor_count, 2);
        assert_eq!(entry_info.predecessor_count, 0);
        
        // Then and else blocks should have entry as predecessor
        let then_info = &analyzer.block_info[1];
        assert_eq!(then_info.predecessor_count, 1);
        assert_eq!(then_info.successor_count, 0);
        
        let else_info = &analyzer.block_info[2];
        assert_eq!(else_info.predecessor_count, 1);
        assert_eq!(else_info.successor_count, 0);
    }
}