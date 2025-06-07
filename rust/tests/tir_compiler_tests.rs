//! Tests for compiling TIR (Test IR) using the TPDE compiler.
//!
//! These tests verify that the compiler components work correctly
//! with TestIR input, similar to the C++ tpde_test executable.

use tpde::test_ir::{TestIR, TestIRAdaptor};
use tpde::core::{Analyzer, IrAdaptor};
use std::fs;
use std::path::Path;

/// Helper to load and parse a TIR file
fn load_tir(filename: &str) -> TestIR {
    let path = Path::new("../tpde/test/filetest/tir").join(filename);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    
    TestIR::parse(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", filename, e))
}

#[test]
fn test_analyzer_rpo() {
    let ir = load_tir("br.tir");
    
    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new();
    
    // Test br1 function
    let funcs: Vec<_> = adaptor.funcs().collect();
    let br1_func = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "br1").unwrap();
    
    // Switch to br1 function in the adaptor first
    adaptor.switch_func(*br1_func);
    
    // Then switch the analyzer to the function
    analyzer.switch_func(&mut adaptor, *br1_func);
    
    // Verify RPO (Reverse Post Order)
    let rpo = analyzer.order();
    assert_eq!(rpo.len(), 3, "br1 should have 3 blocks in RPO");
    
    // The entry block should be first in RPO
    let entry_block = adaptor.entry_block();
    assert_eq!(rpo[0], entry_block, "Entry block should be first in RPO");
}

#[test]
fn test_analyzer_liveness() {
    let ir = load_tir("simple.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new();
    
    // Test myfunc
    let funcs: Vec<_> = adaptor.funcs().collect();
    let myfunc = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "myfunc").unwrap();
    
    // Switch to myfunc in the adaptor first
    adaptor.switch_func(*myfunc);
    
    // Then switch the analyzer to the function
    analyzer.switch_func(&mut adaptor, *myfunc);
    
    // Check that we have liveness info
    let liveness_info = analyzer.liveness(0); // First value
    assert!(liveness_info.is_some(), "Should have liveness info for first value");
}

#[test]
fn test_analyzer_block_layout() {
    let ir = load_tir("br.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    let mut analyzer = Analyzer::new();
    
    // Test condbr1 function
    let funcs: Vec<_> = adaptor.funcs().collect();
    let condbr1 = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "condbr1").unwrap();
    
    // Switch to condbr1 in the adaptor first
    adaptor.switch_func(*condbr1);
    
    // Then switch the analyzer to the function
    analyzer.switch_func(&mut adaptor, *condbr1);
    
    // Get block layout
    let layout = analyzer.order();
    
    // Should have 3 blocks: entry, ret1, ret2
    assert_eq!(layout.len(), 3, "condbr1 should have 3 blocks in layout");
    
    // Entry block should be first
    let entry_block = adaptor.entry_block();
    assert_eq!(layout[0], entry_block, "Entry block should be first in layout");
}

#[test]
fn test_parse_alloca() {
    let ir = load_tir("alloca_align.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    
    // Find the test function
    let funcs: Vec<_> = adaptor.funcs().collect();
    let test_func = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "test").unwrap();
    adaptor.switch_func(*test_func);
    
    // Count allocas
    let allocas = adaptor.cur_static_allocas();
    let alignments: Vec<_> = allocas.iter()
        .map(|&val| adaptor.val_alloca_align(val))
        .collect();
    
    assert_eq!(allocas.len(), 5, "Should have 5 alloca instructions");
    assert_eq!(alignments, vec![1, 2, 4, 8, 16], "Alignments should match expected values");
}

#[test]
fn test_phi_nodes() {
    let ir = load_tir("simple.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    
    // Find myfunc
    let funcs: Vec<_> = adaptor.funcs().collect();
    let myfunc = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "myfunc").unwrap();
    adaptor.switch_func(*myfunc);
    
    // Find the block with PHI node (ret block)
    let blocks: Vec<_> = adaptor.blocks().collect();
    let ret_block = blocks.iter().find(|&&b| {
        adaptor.block_phis(b).count() > 0
    }).unwrap();
    
    // Verify PHI node structure
    let phis: Vec<_> = adaptor.block_phis(*ret_block).collect();
    assert_eq!(phis.len(), 1, "ret block should have 1 PHI node");
    
    let phi_info = adaptor.val_as_phi(phis[0]);
    assert_eq!(phi_info.incoming_count(), 2, "PHI should have 2 incoming values");
}

#[test]
fn test_function_arguments() {
    let ir = load_tir("simple.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    
    // Test secfunc with 2 arguments
    let funcs: Vec<_> = adaptor.funcs().collect();
    let secfunc = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "secfunc").unwrap();
    adaptor.switch_func(*secfunc);
    
    let args: Vec<_> = adaptor.cur_args().collect();
    assert_eq!(args.len(), 2, "secfunc should have 2 arguments");
}

#[test]
fn test_control_flow_graph() {
    let ir = load_tir("br.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    
    // Test br1 function's control flow
    let funcs: Vec<_> = adaptor.funcs().collect();
    let br1_func = funcs.iter().find(|&&f| adaptor.func_link_name(f) == "br1").unwrap();
    adaptor.switch_func(*br1_func);
    
    let blocks: Vec<_> = adaptor.blocks().collect();
    assert_eq!(blocks.len(), 3, "br1 should have 3 blocks");
    
    // Check successors
    let entry_succs: Vec<_> = adaptor.block_succs(blocks[0]).collect();
    assert_eq!(entry_succs.len(), 1, "entry block should have 1 successor");
    
    let sec_succs: Vec<_> = adaptor.block_succs(blocks[1]).collect();
    assert_eq!(sec_succs.len(), 1, "secBlock should have 1 successor");
    
    let ret_succs: Vec<_> = adaptor.block_succs(blocks[2]).collect();
    assert_eq!(ret_succs.len(), 0, "retBlock should have no successors");
}

#[test]
fn test_instruction_operands() {
    let ir = load_tir("add.tir");
    let mut adaptor = TestIRAdaptor::new(&ir);
    
    let funcs: Vec<_> = adaptor.funcs().collect();
    let func = funcs[0];
    adaptor.switch_func(func);
    
    let entry_block = adaptor.entry_block();
    let insts: Vec<_> = adaptor.block_insts(entry_block).collect();
    
    // Should have at least the add instruction and terminate
    assert!(insts.len() >= 2, "Should have at least 2 instructions");
    
    // Find the add instruction (should have 2 operands)
    for inst in &insts {
        let operands: Vec<_> = adaptor.inst_operands(*inst).collect();
        let results: Vec<_> = adaptor.inst_results(*inst).collect();
        
        if operands.len() == 2 {
            // This should be the add instruction
            assert_eq!(results.len(), 1, "Add should produce 1 result");
        }
    }
}

#[test]
fn test_basic_tir_parsing() {
    // Test that we can parse all the TIR test files
    let test_files = [
        "add.tir",
        "alloca_align.tir", 
        "br.tir",
        "call.tir",
        "duplicate_funcs.tir",
        "func.tir",
        "simple.tir",
    ];
    
    for file in &test_files {
        let ir = load_tir(file);
        assert!(!ir.functions.is_empty(), "{} should have functions", file);
    }
}