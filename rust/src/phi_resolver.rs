//! PHI node resolution for SSA form.
//!
//! This module implements PHI node resolution with cycle detection,
//! based on the C++ implementation but using arena allocation.

use crate::compilation_session::CompilationSession;
use crate::error::{CompileError, CompileResult};
use crate::function_analyzer::FunctionAnalyzer;
use crate::llvm_compiler_concrete::LlvmCompiler;
use crate::register_file::AsmReg;


/// PHI resolution state for tracking cycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhiState {
    /// Not yet visited.
    Unvisited,
    /// Currently being resolved (on the stack).
    OnStack,
    /// Resolution complete.
    Resolved,
}

/// PHI resolution move operation.
#[derive(Debug, Clone, Copy)]
pub struct PhiMove {
    /// Source value index.
    pub src_value: usize,
    /// Source register (if already in register).
    pub src_reg: Option<AsmReg>,
    /// Destination value index.
    pub dst_value: usize,
    /// Destination register (if already allocated).
    pub dst_reg: Option<AsmReg>,
    /// Whether this move is part of a cycle.
    pub in_cycle: bool,
}

/// PHI node resolver that handles cycle detection and resolution ordering.
pub struct PhiResolver<'ctx, 'arena> {
    /// Reference to the compilation session for arena allocation.
    session: &'arena CompilationSession<'arena>,
    
    /// Function analyzer with PHI node information.
    analyzer: &'arena FunctionAnalyzer<'ctx, 'arena>,
    
    /// PHI resolution state for each PHI node.
    phi_states: &'arena mut [PhiState],
    
    /// Moves to perform for PHI resolution.
    moves: &'arena mut [PhiMove],
    
    /// Number of moves.
    move_count: usize,
    
    /// Temporary scratch space for cycle breaking.
    /// TODO: Implement cycle detection and breaking using temp registers
    #[allow(dead_code)]
    temp_regs: &'arena mut [Option<AsmReg>],
}

impl<'ctx, 'arena> PhiResolver<'ctx, 'arena> {
    /// Maximum number of PHI nodes we support.
    const MAX_PHI_NODES: usize = 128;
    
    /// Maximum number of moves.
    const MAX_MOVES: usize = 256;
    
    /// Create a new PHI resolver.
    pub fn new(
        session: &'arena CompilationSession<'arena>,
        analyzer: &'arena FunctionAnalyzer<'ctx, 'arena>,
    ) -> CompileResult<Self> {
        // Allocate state arrays in the arena
        let phi_states = session.alloc_slice(&vec![PhiState::Unvisited; Self::MAX_PHI_NODES]);
        let phi_states = unsafe {
            std::slice::from_raw_parts_mut(phi_states.as_ptr() as *mut PhiState, phi_states.len())
        };
        
        let moves = session.alloc_slice(&vec![PhiMove {
            src_value: 0,
            src_reg: None,
            dst_value: 0,
            dst_reg: None,
            in_cycle: false,
        }; Self::MAX_MOVES]);
        let moves = unsafe {
            std::slice::from_raw_parts_mut(moves.as_ptr() as *mut PhiMove, moves.len())
        };
        
        let temp_regs = session.alloc_slice(&vec![None::<AsmReg>; 16]); // Max temp registers
        let temp_regs = unsafe {
            std::slice::from_raw_parts_mut(temp_regs.as_ptr() as *mut Option<AsmReg>, temp_regs.len())
        };
        
        Ok(Self {
            session,
            analyzer,
            phi_states,
            moves,
            move_count: 0,
            temp_regs,
        })
    }
    
    /// Resolve PHI nodes for a specific block transition.
    pub fn resolve_phis_for_edge(
        &mut self,
        compiler: &mut LlvmCompiler<'ctx, 'arena>,
        from_block_idx: usize,
        to_block_idx: usize,
    ) -> CompileResult<()> {
        // Reset state
        self.move_count = 0;
        for state in &mut self.phi_states[..Self::MAX_PHI_NODES] {
            *state = PhiState::Unvisited;
        }
        
        // Collect all PHI nodes in the target block
        let phi_nodes = self.analyzer.get_phi_nodes(to_block_idx);
        
        // Build move list for this edge
        for (phi_idx, phi_info) in phi_nodes.iter().enumerate() {
            // Find incoming value for this edge
            let incoming_values = self.analyzer.get_phi_incoming(phi_idx);
            
            for incoming in incoming_values {
                if incoming.pred_block_idx == from_block_idx {
                    // Add move from incoming value to PHI result
                    if self.move_count >= Self::MAX_MOVES {
                        return Err(CompileError::BlockLayout {
                            reason: "Too many PHI moves".to_string(),
                        });
                    }
                    
                    self.moves[self.move_count] = PhiMove {
                        src_value: incoming.value_idx,
                        src_reg: None, // Will be filled during resolution
                        dst_value: phi_info.result_idx,
                        dst_reg: None, // Will be filled during resolution
                        in_cycle: false,
                    };
                    self.move_count += 1;
                    break;
                }
            }
        }
        
        // Detect cycles using DFS
        self.detect_cycles()?;
        
        // Resolve moves in correct order
        self.emit_moves(compiler)?;
        
        Ok(())
    }
    
    /// Detect cycles in the move graph using DFS.
    fn detect_cycles(&mut self) -> CompileResult<()> {
        // Build adjacency information: for each destination, find its source
        let dest_to_src = self.session.alloc_slice(&vec![None::<usize>; Self::MAX_MOVES]);
        let dest_to_src = unsafe {
            std::slice::from_raw_parts_mut(dest_to_src.as_ptr() as *mut Option<usize>, dest_to_src.len())
        };
        
        for i in 0..self.move_count {
            let dst = self.moves[i].dst_value;
            // Simple mapping - in real implementation would use proper lookup
            if dst < Self::MAX_MOVES {
                dest_to_src[dst] = Some(i);
            }
        }
        
        // Visit each move to detect cycles
        for start_idx in 0..self.move_count {
            if self.phi_states[start_idx] == PhiState::Unvisited {
                self.dfs_cycle_detect(start_idx, dest_to_src)?;
            }
        }
        
        Ok(())
    }
    
    /// DFS for cycle detection.
    fn dfs_cycle_detect(
        &mut self,
        move_idx: usize,
        dest_to_src: &[Option<usize>],
    ) -> CompileResult<()> {
        self.phi_states[move_idx] = PhiState::OnStack;
        
        // Check if this move's source is the destination of another move
        let src_value = self.moves[move_idx].src_value;
        if src_value < dest_to_src.len() {
            if let Some(next_idx) = dest_to_src[src_value] {
                match self.phi_states[next_idx] {
                    PhiState::Unvisited => {
                        self.dfs_cycle_detect(next_idx, dest_to_src)?;
                    }
                    PhiState::OnStack => {
                        // Cycle detected! Mark all moves in the cycle
                        self.mark_cycle(move_idx, next_idx);
                    }
                    PhiState::Resolved => {
                        // Already processed
                    }
                }
            }
        }
        
        self.phi_states[move_idx] = PhiState::Resolved;
        Ok(())
    }
    
    /// Mark moves as part of a cycle.
    fn mark_cycle(&mut self, start_idx: usize, end_idx: usize) {
        // In a real implementation, we'd walk the cycle and mark all moves
        // For now, just mark the detected moves
        self.moves[start_idx].in_cycle = true;
        self.moves[end_idx].in_cycle = true;
    }
    
    /// Emit moves in the correct order, handling cycles.
    fn emit_moves(&mut self, compiler: &mut LlvmCompiler<'ctx, 'arena>) -> CompileResult<()> {
        // First, emit all non-cycle moves
        for i in 0..self.move_count {
            if !self.moves[i].in_cycle {
                self.emit_single_move(compiler, i)?;
            }
        }
        
        // Then handle cycles by breaking them with temporary registers
        let cycle_handled = self.session.alloc_slice(&vec![false; Self::MAX_MOVES]);
        let cycle_handled = unsafe {
            std::slice::from_raw_parts_mut(cycle_handled.as_ptr() as *mut bool, cycle_handled.len())
        };
        
        for i in 0..self.move_count {
            if self.moves[i].in_cycle && !cycle_handled[i] {
                self.break_cycle_at(compiler, i, cycle_handled)?;
            }
        }
        
        Ok(())
    }
    
    /// Emit a single move operation.
    fn emit_single_move(
        &mut self,
        _compiler: &mut LlvmCompiler<'ctx, 'arena>,
        move_idx: usize,
    ) -> CompileResult<()> {
        let mv = &self.moves[move_idx];
        
        // In a real implementation, we'd:
        // 1. Get or allocate registers for source and destination
        // 2. Generate appropriate move instructions
        // 3. Update value tracking
        
        // For now, just log the move
        log::trace!("   PHI   PHI move: v{} -> v{} {}", 
                 mv.src_value, 
                 mv.dst_value,
                 if mv.in_cycle { "(in cycle)" } else { "" });
        
        Ok(())
    }
    
    /// Break a cycle by using a temporary register.
    fn break_cycle_at(
        &mut self,
        compiler: &mut LlvmCompiler<'ctx, 'arena>,
        start_idx: usize,
        handled: &mut [bool],
    ) -> CompileResult<()> {
        // Find an available temporary register
        let temp_reg = self.allocate_temp_reg()?;
        
        // Save the start value to temp
        log::debug!("Breaking cycle: saving v{} to temp", 
                 self.moves[start_idx].src_value);
        
        // Emit moves in cycle order
        let mut current_idx = start_idx;
        loop {
            handled[current_idx] = true;
            
            // Find next move in cycle
            let next_idx = self.find_next_in_cycle(current_idx);
            if next_idx == start_idx {
                // Cycle complete - restore from temp
                log::debug!("Cycle complete: restoring from temp to v{}", 
                         self.moves[current_idx].dst_value);
                break;
            }
            
            self.emit_single_move(compiler, current_idx)?;
            current_idx = next_idx;
        }
        
        // Release temporary register
        self.release_temp_reg(temp_reg);
        
        Ok(())
    }
    
    /// Find the next move in a cycle.
    fn find_next_in_cycle(&self, current_idx: usize) -> usize {
        let current_src = self.moves[current_idx].src_value;
        
        // Find move whose destination is our source
        for i in 0..self.move_count {
            if self.moves[i].dst_value == current_src && self.moves[i].in_cycle {
                return i;
            }
        }
        
        // Should not happen if cycle detection is correct
        current_idx
    }
    
    /// Allocate a temporary register for cycle breaking.
    fn allocate_temp_reg(&mut self) -> CompileResult<AsmReg> {
        // In a real implementation, we'd get this from the register allocator
        // For now, use a fixed scratch register
        Ok(AsmReg::new(0, 11)) // R11 as scratch
    }
    
    /// Release a temporary register.
    fn release_temp_reg(&mut self, _reg: AsmReg) {
        // In a real implementation, we'd return this to the register allocator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;
    use bumpalo::Bump;
    
    #[test]
    fn test_phi_cycle_detection() {
        // This test would create a function with PHI nodes that form a cycle
        // and verify that the resolver correctly detects and handles it
        
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();
        
        // Create a function with a loop containing PHI nodes
        let fn_type = i32_type.fn_type(&[], false);
        let function = module.add_function("test_phi", fn_type, None);
        
        let entry = context.append_basic_block(function, "entry");
        let loop_header = context.append_basic_block(function, "loop");
        let loop_body = context.append_basic_block(function, "body");
        let exit = context.append_basic_block(function, "exit");
        
        let builder = context.create_builder();
        
        // Entry block
        builder.position_at_end(entry);
        let initial1 = i32_type.const_int(1, false);
        let initial2 = i32_type.const_int(2, false);
        builder.build_unconditional_branch(loop_header).unwrap();
        
        // Loop header with PHI nodes
        builder.position_at_end(loop_header);
        let phi1 = builder.build_phi(i32_type, "i").unwrap();
        let phi2 = builder.build_phi(i32_type, "j").unwrap();
        
        // Add incoming values from entry
        phi1.add_incoming(&[(&initial1, entry)]);
        phi2.add_incoming(&[(&initial2, entry)]);
        
        // Loop body - swap values (creates cycle)
        builder.position_at_end(loop_body);
        let temp1 = phi2.as_basic_value().into_int_value();
        let temp2 = phi1.as_basic_value().into_int_value();
        builder.build_unconditional_branch(loop_header).unwrap();
        
        // Add incoming values from loop body (creates cycle: i <- j, j <- i)
        phi1.add_incoming(&[(&temp1, loop_body)]);
        phi2.add_incoming(&[(&temp2, loop_body)]);
        
        // Exit
        builder.position_at_end(exit);
        builder.build_return(Some(&phi1.as_basic_value())).unwrap();
        
        // Create arena and session
        let arena = Bump::new();
        let session = CompilationSession::new(&arena);
        
        // Analyze function
        let analyzer = FunctionAnalyzer::analyze(&session, function).unwrap();
        
        // Create PHI resolver
        let _resolver = PhiResolver::new(&session, &analyzer).unwrap();
        
        // The resolver should detect the cycle between phi1 and phi2
        // In a full implementation, we'd verify the cycle detection
        // and proper move ordering
    }
}