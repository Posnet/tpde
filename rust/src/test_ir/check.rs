//! FileCheck-style test validation for TIR files.
//!
//! This module provides functionality to parse CHECK directives from TIR files
//! and validate output against expected patterns, similar to LLVM's FileCheck tool
//! but implemented in a Rust-native way.

use super::{TestIR, TestIRAdaptor};
use crate::core::{Analyzer, IrAdaptor};
use std::collections::VecDeque;

/// A CHECK directive extracted from a TIR file
#[derive(Debug, Clone)]
pub enum CheckDirective {
    /// CHECK: pattern - Match exact pattern
    Check(String),
    /// CHECK-LABEL: pattern - Label for a section
    CheckLabel(String),
    /// CHECK-NEXT: pattern - Match on the next line
    CheckNext(String),
    /// CHECK-EMPTY - Match empty line
    CheckEmpty,
    /// COM: comment - Comment, ignored
    Comment(String),
}

/// A RUN directive specifying how to execute the test
#[derive(Debug, Clone)]
pub struct RunDirective {
    pub command: String,
    pub args: Vec<String>,
}

/// Test specification extracted from a TIR file
#[derive(Debug)]
pub struct TestSpec {
    pub run_directives: Vec<RunDirective>,
    pub check_directives: Vec<CheckDirective>,
    pub tir_content: String,
}

impl TestSpec {
    /// Parse a TIR file to extract test specifications
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut run_directives = Vec::new();
        let mut check_directives = Vec::new();
        let mut tir_lines = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("; RUN:") {
                // Parse RUN directive
                let run_cmd = trimmed.strip_prefix("; RUN:").unwrap().trim();
                let parts: Vec<&str> = run_cmd.split_whitespace().collect();
                if !parts.is_empty() {
                    run_directives.push(RunDirective {
                        command: parts[0].to_string(),
                        args: parts[1..].iter().map(|s| s.to_string()).collect(),
                    });
                }
            } else if trimmed.starts_with("; CHECK-LABEL:") {
                let pattern = trimmed.strip_prefix("; CHECK-LABEL:").unwrap().trim();
                check_directives.push(CheckDirective::CheckLabel(pattern.to_string()));
            } else if trimmed.starts_with("; CHECK-NEXT:") {
                let pattern = trimmed.strip_prefix("; CHECK-NEXT:").unwrap().trim();
                check_directives.push(CheckDirective::CheckNext(pattern.to_string()));
            } else if trimmed.starts_with("; CHECK-EMPTY") {
                check_directives.push(CheckDirective::CheckEmpty);
            } else if trimmed.starts_with("; CHECK:") {
                let pattern = trimmed.strip_prefix("; CHECK:").unwrap().trim();
                check_directives.push(CheckDirective::Check(pattern.to_string()));
            } else if trimmed.starts_with("; COM:") {
                let comment = trimmed.strip_prefix("; COM:").unwrap().trim();
                check_directives.push(CheckDirective::Comment(comment.to_string()));
            } else {
                // Regular TIR content
                tir_lines.push(line);
            }
        }

        Ok(TestSpec {
            run_directives,
            check_directives,
            tir_content: tir_lines.join("\n"),
        })
    }
}

/// Test runner that executes TIR tests
pub struct TestRunner {
    verbose: bool,
}

impl TestRunner {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }

    /// Run a TIR test and validate output
    pub fn run_test(&self, spec: &TestSpec) -> Result<(), String> {
        // Parse the TIR content
        let ir = TestIR::parse(&spec.tir_content)?;

        // Execute based on run directives
        for run_dir in &spec.run_directives {
            let output = self.execute_command(&ir, run_dir)?;
            self.validate_output(&output, &spec.check_directives)?;
        }

        Ok(())
    }

    /// Execute a test command and return the output
    fn execute_command(&self, ir: &TestIR, run_dir: &RunDirective) -> Result<String, String> {
        // Parse command arguments
        let mut print_ir = false;
        let mut print_rpo = false;
        let mut print_liveness = false;
        let mut run_until = "codegen";

        for (i, arg) in run_dir.args.iter().enumerate() {
            match arg.as_str() {
                "--print-ir" => print_ir = true,
                "--print-rpo" => print_rpo = true,
                "--print-liveness" => print_liveness = true,
                "--run-until" => {
                    if i + 1 < run_dir.args.len() {
                        run_until = &run_dir.args[i + 1];
                    }
                }
                _ => {}
            }
        }

        let mut output = Vec::new();

        if print_ir {
            output.push("Printing IR".to_string());
            output.push(format!("{}", ir));
        }

        if print_rpo || print_liveness || run_until == "analyzer" {
            let mut adaptor = TestIRAdaptor::new(ir);
            let mut analyzer = Analyzer::new();

            for func in adaptor.funcs() {
                let func_name = adaptor.func_link_name(func).to_string(); // Clone the name

                analyzer.switch_func(&mut adaptor, func);

                if print_rpo {
                    output.push(format!("RPO for func {}", func_name));
                    for (idx, block) in analyzer.order().iter().enumerate() {
                        let block_name = adaptor.block_name(*block);
                        output.push(format!("{}: {}", idx, block_name));
                    }
                    output.push("End RPO".to_string());
                }

                if print_liveness {
                    output.push(format!("Liveness for {}", func_name));

                    // Get block names for index mapping
                    let blocks: Vec<_> = analyzer.order().to_vec();
                    let block_names: Vec<String> = blocks
                        .iter()
                        .map(|&b| adaptor.block_name(b).to_string())
                        .collect();

                    // Print liveness info for each value
                    let mut value_idx = 0;

                    // First, function arguments
                    for _arg in adaptor.cur_args() {
                        if let Some(liveness) = analyzer.liveness(value_idx) {
                            let first_block = block_names
                                .get(liveness.first)
                                .map(|s| s.as_str())
                                .unwrap_or("?");
                            let last_block = block_names
                                .get(liveness.last)
                                .map(|s| s.as_str())
                                .unwrap_or("?");

                            output.push(format!(
                                "{}: {} refs, {}->{} ({}->{}) , lf: {}",
                                value_idx,
                                liveness.ref_count,
                                liveness.first,
                                liveness.last,
                                first_block,
                                last_block,
                                liveness.last_full
                            ));
                        } else {
                            output.push(format!("{}: ignored", value_idx));
                        }
                        value_idx += 1;
                    }

                    // Then, values in blocks
                    for block in &blocks {
                        // PHIs
                        for _phi in adaptor.block_phis(*block) {
                            if let Some(liveness) = analyzer.liveness(value_idx) {
                                let first_block = block_names
                                    .get(liveness.first)
                                    .map(|s| s.as_str())
                                    .unwrap_or("?");
                                let last_block = block_names
                                    .get(liveness.last)
                                    .map(|s| s.as_str())
                                    .unwrap_or("?");

                                output.push(format!(
                                    "{}: {} refs, {}->{}({}->{}) , lf: {}",
                                    value_idx,
                                    liveness.ref_count,
                                    liveness.first,
                                    liveness.last,
                                    first_block,
                                    last_block,
                                    liveness.last_full
                                ));
                            } else {
                                output.push(format!("{}: ignored", value_idx));
                            }
                            value_idx += 1;
                        }

                        // Instructions
                        for _inst in adaptor.block_insts(*block) {
                            if let Some(liveness) = analyzer.liveness(value_idx) {
                                let first_block = block_names
                                    .get(liveness.first)
                                    .map(|s| s.as_str())
                                    .unwrap_or("?");
                                let last_block = block_names
                                    .get(liveness.last)
                                    .map(|s| s.as_str())
                                    .unwrap_or("?");

                                output.push(format!(
                                    "{}: {} refs, {}->{}({}->{}) , lf: {}",
                                    value_idx,
                                    liveness.ref_count,
                                    liveness.first,
                                    liveness.last,
                                    first_block,
                                    last_block,
                                    liveness.last_full
                                ));
                            } else {
                                output.push(format!("{}: ignored", value_idx));
                            }
                            value_idx += 1;
                        }
                    }

                    output.push("End Liveness".to_string());
                }
            }
        }

        Ok(output.join("\n"))
    }

    /// Validate output against CHECK directives
    pub fn validate_output(
        &self,
        output: &str,
        directives: &[CheckDirective],
    ) -> Result<(), String> {
        let output_lines: VecDeque<_> = output.lines().collect();
        let mut line_idx = 0;

        for directive in directives {
            match directive {
                CheckDirective::Comment(_) => continue,

                CheckDirective::Check(pattern) => {
                    let found = output_lines
                        .iter()
                        .skip(line_idx)
                        .position(|line| line.contains(pattern));

                    match found {
                        Some(idx) => {
                            line_idx += idx + 1; // Move to the next line after the match
                            if self.verbose {
                                println!("CHECK: '{}' found at line {}", pattern, line_idx - 1);
                            }
                        }
                        None => {
                            return Err(format!(
                                "CHECK: pattern '{}' not found in output",
                                pattern
                            ));
                        }
                    }
                }

                CheckDirective::CheckLabel(pattern) => {
                    let found = output_lines
                        .iter()
                        .skip(line_idx)
                        .position(|line| line.contains(pattern));

                    match found {
                        Some(idx) => {
                            line_idx += idx + 1;
                            if self.verbose {
                                println!(
                                    "CHECK-LABEL: '{}' found at line {}",
                                    pattern,
                                    line_idx - 1
                                );
                            }
                        }
                        None => {
                            return Err(format!("CHECK-LABEL: pattern '{}' not found", pattern));
                        }
                    }
                }

                CheckDirective::CheckNext(pattern) => {
                    if line_idx >= output_lines.len() {
                        return Err(format!("CHECK-NEXT: no more lines, expected '{}'", pattern));
                    }

                    let line = output_lines[line_idx];
                    if !line.contains(pattern) {
                        return Err(format!(
                            "CHECK-NEXT: expected '{}' but got '{}'",
                            pattern, line
                        ));
                    }

                    if self.verbose {
                        println!("CHECK-NEXT: '{}' matches at line {}", pattern, line_idx);
                    }
                    line_idx += 1;
                }

                CheckDirective::CheckEmpty => {
                    if line_idx >= output_lines.len() {
                        continue; // End of output counts as empty
                    }

                    let line = output_lines[line_idx];
                    if !line.trim().is_empty() {
                        return Err(format!(
                            "CHECK-EMPTY: expected empty line but got '{}'",
                            line
                        ));
                    }

                    if self.verbose {
                        println!("CHECK-EMPTY: matches at line {}", line_idx);
                    }
                    line_idx += 1;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_directives() {
        let content = r#"; RUN: %tpde_test --print-ir %s
; CHECK: Printing IR
; CHECK-LABEL: Function test
; CHECK-NEXT: Block entry
; COM: This is a comment
test() {
  entry:
    terminate
}"#;

        let spec = TestSpec::parse(content).unwrap();
        assert_eq!(spec.run_directives.len(), 1);
        assert_eq!(spec.check_directives.len(), 4);
        assert!(spec.tir_content.contains("test()"));
    }

    #[test]
    fn test_check_matching() {
        let runner = TestRunner::new(false);
        let output = "Printing IR\nFunction test\nBlock entry\n";

        let directives = vec![
            CheckDirective::Check("Printing IR".to_string()),
            CheckDirective::CheckLabel("Function test".to_string()),
            CheckDirective::CheckNext("Block entry".to_string()),
        ];

        runner.validate_output(output, &directives).unwrap();
    }

    #[test]
    fn test_check_next_failure() {
        let runner = TestRunner::new(false);
        let output = "Line 1\nLine 2\nLine 3\n";

        let directives = vec![
            CheckDirective::Check("Line 1".to_string()),
            CheckDirective::CheckNext("Line 3".to_string()), // Should fail
        ];

        let result = runner.validate_output(output, &directives);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("CHECK-NEXT"));
    }
}
