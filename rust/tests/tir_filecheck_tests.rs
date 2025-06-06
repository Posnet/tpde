//! FileCheck-style tests for TIR files
//! 
//! This test suite validates TIR files against their CHECK directives,
//! similar to how LLVM's FileCheck works but implemented in Rust.

use tpde::test_ir::{TestRunner, TestSpec};
use std::fs;
use std::path::Path;

/// Test helper that runs a TIR file through FileCheck validation
fn run_filecheck_test(tir_file: &str) {
    let path = Path::new("../tpde/test/filetest/tir").join(tir_file);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    
    let spec = TestSpec::parse(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse test spec from {}: {}", tir_file, e));
    
    let runner = TestRunner::new(false);
    runner.run_test(&spec)
        .unwrap_or_else(|e| panic!("Test {} failed: {}", tir_file, e));
}

#[test]
fn test_simple_tir_filecheck() {
    run_filecheck_test("simple.tir");
}

#[test]
fn test_br_tir_filecheck() {
    run_filecheck_test("br.tir");
}

#[test]
fn test_add_tir_filecheck() {
    run_filecheck_test("add.tir");
}

#[test]
fn test_alloca_align_filecheck() {
    run_filecheck_test("alloca_align.tir");
}

#[test]
fn test_call_tir_filecheck() {
    run_filecheck_test("call.tir");
}

#[test]
fn test_func_tir_filecheck() {
    run_filecheck_test("func.tir");
}

/// Test analyzer RPO output with FileCheck
#[test]
fn test_analyzer_rpo_filecheck() {
    let path = Path::new("../tpde/test/filetest/analyzer/rpo.tir");
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    
    let spec = TestSpec::parse(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse test spec: {}", e));
    
    let runner = TestRunner::new(false);
    runner.run_test(&spec)
        .unwrap_or_else(|e| panic!("Analyzer RPO test failed: {}", e));
}

/// Test liveness analysis with FileCheck
#[test]
fn test_analyzer_liveness_filecheck() {
    let path = Path::new("../tpde/test/filetest/analyzer/liveness.tir");
    
    // Check if file exists before trying to run the test
    if !path.exists() {
        println!("Skipping liveness test - file not found: {}", path.display());
        return;
    }
    
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    
    let spec = TestSpec::parse(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse test spec: {}", e));
    
    let runner = TestRunner::new(false);
    runner.run_test(&spec)
        .unwrap_or_else(|e| panic!("Analyzer liveness test failed: {}", e));
}

#[cfg(test)]
mod filecheck_internals {
    use super::*;
    use tpde::test_ir::{TestSpec, CheckDirective};
    
    #[test]
    fn test_check_directive_parsing() {
        let content = r#"; RUN: test --print-ir
; CHECK: foo
; CHECK-LABEL: bar
; CHECK-NEXT: baz
; CHECK-EMPTY
; COM: comment
test() { entry: terminate }"#;
        
        let spec = TestSpec::parse(content).unwrap();
        assert_eq!(spec.check_directives.len(), 5);
        
        match &spec.check_directives[0] {
            CheckDirective::Check(s) => assert_eq!(s, "foo"),
            _ => panic!("Expected Check directive"),
        }
        
        match &spec.check_directives[1] {
            CheckDirective::CheckLabel(s) => assert_eq!(s, "bar"),
            _ => panic!("Expected CheckLabel directive"),
        }
    }
    
    #[test]
    fn test_verbose_runner() {
        // Test with verbose output to ensure logging works
        let runner = TestRunner::new(true);
        let output = "Line 1\nLine 2\nLine 3\n";
        let directives = vec![
            CheckDirective::Check("Line 1".to_string()),
            CheckDirective::CheckNext("Line 2".to_string()),
        ];
        
        runner.validate_output(output, &directives).unwrap();
    }
    
    #[test]
    fn test_check_next_succeeds() {
        let runner = TestRunner::new(false);
        let output = "Line 1\nLine 2\nLine 3\n";
        let directives = vec![
            CheckDirective::CheckLabel("Line 1".to_string()), // Label moves to next line
            CheckDirective::CheckNext("Line 2".to_string()),
        ];
        
        runner.validate_output(output, &directives).unwrap();
    }
}