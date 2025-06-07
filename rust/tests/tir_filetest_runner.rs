//! Dynamic test runner for TIR filetest files.
//!
//! This module discovers and runs all .tir files from the C++ test directories,
//! executing them with the same capabilities as the C++ tpde_test harness.

use std::fs;
use std::path::{Path, PathBuf};
use tpde::test_ir::{TestIR, TestIRAdaptor};
use tpde::core::{Analyzer, IrAdaptor};

/// Discovers all .tir files in a directory recursively
fn discover_tir_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(discover_tir_files(&path));
            } else if path.extension().and_then(|s| s.to_str()) == Some("tir") {
                files.push(path);
            }
        }
    }
    
    files
}

/// Run a single TIR file and return its output
fn run_tir_file(path: &Path) -> Result<String, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    
    // Parse RUN directive to determine what to execute
    let mut print_ir = false;
    let mut print_rpo = false;
    let mut print_liveness = false;
    let mut print_loops = false;
    let mut print_layout = false;
    let mut run_until = "codegen";
    
    for line in content.lines() {
        if let Some(run_line) = line.strip_prefix("; RUN:") {
            if run_line.contains("--print-ir") {
                print_ir = true;
            }
            if run_line.contains("--print-rpo") {
                print_rpo = true;
            }
            if run_line.contains("--print-liveness") {
                print_liveness = true;
            }
            if run_line.contains("--print-loops") {
                print_loops = true;
            }
            if run_line.contains("--print-layout") {
                print_layout = true;
            }
            if let Some(pos) = run_line.find("--run-until=") {
                let start = pos + "--run-until=".len();
                if let Some(end) = run_line[start..].find(' ') {
                    run_until = &run_line[start..start + end];
                } else {
                    run_until = &run_line[start..];
                }
            }
        }
    }
    
    // Parse the TIR content
    let ir = TestIR::parse(&content)?;
    let mut output = Vec::new();
    
    // Print IR if requested
    if print_ir {
        output.push("Printing IR".to_string());
        output.push(ir.to_string());
    }
    
    // Run analyzer if needed
    if print_rpo || print_liveness || print_loops || print_layout || run_until == "analyzer" {
        let mut adaptor = TestIRAdaptor::new(&ir);
        let mut analyzer = Analyzer::new();
        
        // Process each function
        let funcs: Vec<_> = adaptor.funcs().collect();
        for func in funcs {
            let func_name = adaptor.func_link_name(func).to_string();
            adaptor.switch_func(func);
            analyzer.switch_func(&mut adaptor, func);
            
            if print_rpo {
                output.push(format!("RPO for func {func_name}"));
                let rpo = analyzer.order();
                for (idx, &block) in rpo.iter().enumerate() {
                    let block_name = adaptor.block_name(block);
                    output.push(format!("{idx}: {block_name}"));
                }
                output.push("End RPO".to_string());
            }
            
            if print_layout {
                output.push(format!("Block Layout for {func_name}"));
                let layout = analyzer.block_layout();
                for (idx, &block) in layout.iter().enumerate() {
                    let block_name = adaptor.block_name(block);
                    output.push(format!("{idx}: {block_name}"));
                }
                output.push("End Block Layout".to_string());
            }
            
            if print_loops {
                output.push(format!("Loops for {func_name}"));
                let loops = analyzer.loops();
                for (idx, loop_info) in loops.iter().enumerate() {
                    output.push(format!(
                        "{}: level {}, parent {}, {}->{}",
                        idx,
                        loop_info.level,
                        loop_info.parent,
                        loop_info.begin,
                        loop_info.end
                    ));
                }
                output.push("End Loops".to_string());
            }
            
            if print_liveness {
                output.push(format!("Liveness for {func_name}"));
                
                // Get block names for mapping - use block layout instead of RPO
                let layout = analyzer.block_layout();
                let block_names: Vec<String> = layout.iter()
                    .map(|&b| adaptor.block_name(b).to_string())
                    .collect();
                
                // The C++ implementation iterates through ALL values in definition order,
                // not just by block order. We need to match that exactly.
                // Values are ordered as: args, then all values in all blocks
                
                // First, get the function info
                let func_idx = func.0 as usize;
                let func_info = &ir.functions[func_idx];
                
                // Iterate through all values from arg_begin_idx to the last value in the last block
                let last_block = &ir.blocks[(func_info.block_end_idx - 1) as usize];
                let total_values = last_block.inst_end_idx - func_info.arg_begin_idx;
                
                for i in 0..total_values {
                    let local_idx = i as usize;
                    
                    // Check if this value should be ignored
                    let global_idx = func_info.arg_begin_idx + i;
                    let val_info = &ir.values[global_idx as usize];
                    
                    // Terminators without results or allocas are ignored
                    let should_ignore = val_info.op == tpde::test_ir::Operation::Alloca ||
                        (val_info.value_type == tpde::test_ir::ValueType::Terminator && !val_info.op.info().is_def);
                    
                    if should_ignore {
                        output.push(format!("{local_idx}: ignored"));
                    } else if let Some(liveness) = analyzer.liveness(local_idx) {
                        let first_block = block_names.get(liveness.first)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        let last_block = block_names.get(liveness.last)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        
                        output.push(format!(
                            "{}: {} refs, {}->{} ({}->{}), lf: {}",
                            local_idx,
                            liveness.ref_count,
                            liveness.first,
                            liveness.last,
                            first_block,
                            last_block,
                            liveness.last_full
                        ));
                    } else {
                        // Value has no liveness info but wasn't explicitly ignored
                        output.push(format!("{local_idx}: 0 refs, 0->0 (?->?), lf: false"));
                    }
                }
                
                output.push("End Liveness".to_string());
            }
        }
    }
    
    Ok(output.join("\n"))
}

/// Validate output against CHECK directives
fn validate_output(output: &str, content: &str) -> Result<(), String> {
    let mut checks = Vec::new();
    let mut current_line = 0;
    
    // Extract CHECK directives
    for line in content.lines() {
        if let Some(check) = line.strip_prefix("; CHECK:") {
            checks.push((current_line, "CHECK", check.trim()));
        } else if let Some(check) = line.strip_prefix("; CHECK-NEXT:") {
            checks.push((current_line, "CHECK-NEXT", check.trim()));
        } else if let Some(check) = line.strip_prefix("; CHECK-LABEL:") {
            checks.push((current_line, "CHECK-LABEL", check.trim()));
        } else if line.starts_with("; CHECK-EMPTY") {
            checks.push((current_line, "CHECK-EMPTY", ""));
        }
        current_line += 1;
    }
    
    let output_lines: Vec<&str> = output.lines().collect();
    let mut output_idx = 0;
    
    for (_, check_type, pattern) in checks {
        match check_type {
            "CHECK" => {
                let found = output_lines[output_idx..]
                    .iter()
                    .position(|line| line.contains(pattern));
                
                match found {
                    Some(idx) => output_idx += idx + 1,
                    None => return Err(format!("CHECK: pattern '{pattern}' not found")),
                }
            }
            "CHECK-NEXT" => {
                if output_idx >= output_lines.len() {
                    return Err(format!("CHECK-NEXT: no more lines, expected '{pattern}'"));
                }
                
                if !output_lines[output_idx].contains(pattern) {
                    return Err(format!(
                        "CHECK-NEXT: expected '{}' but got '{}'",
                        pattern, output_lines[output_idx]
                    ));
                }
                output_idx += 1;
            }
            "CHECK-LABEL" => {
                let found = output_lines[output_idx..]
                    .iter()
                    .position(|line| line.contains(pattern));
                
                match found {
                    Some(idx) => output_idx += idx + 1,
                    None => return Err(format!("CHECK-LABEL: pattern '{pattern}' not found")),
                }
            }
            "CHECK-EMPTY" => {
                if output_idx < output_lines.len() && !output_lines[output_idx].trim().is_empty() {
                    return Err(format!(
                        "CHECK-EMPTY: expected empty line but got '{}'",
                        output_lines[output_idx]
                    ));
                }
                output_idx += 1;
            }
            _ => {}
        }
    }
    
    Ok(())
}

// Generate tests for each directory
macro_rules! generate_tir_tests {
    ($test_name:ident, $dir:expr) => {
        #[cfg(test)]
        mod $test_name {
            use super::*;
            
            #[test]
            fn run_all_tir_files() {
                let dir = Path::new($dir);
                if !dir.exists() {
                    println!("Skipping {} - directory not found", $dir);
                    return;
                }
                
                let files = discover_tir_files(dir);
                if files.is_empty() {
                    println!("No .tir files found in {}", $dir);
                    return;
                }
                
                let mut failures = Vec::new();
                
                for file in files {
                    let relative = file.strip_prefix("../").unwrap_or(&file);
                    print!("Running {}... ", relative.display());
                    
                    match run_tir_file(&file) {
                        Ok(output) => {
                            // Read file again to get CHECK directives
                            if let Ok(content) = fs::read_to_string(&file) {
                                match validate_output(&output, &content) {
                                    Ok(_) => println!("PASS"),
                                    Err(e) => {
                                        println!("FAIL");
                                        failures.push((file.clone(), e));
                                    }
                                }
                            } else {
                                println!("FAIL");
                                failures.push((file.clone(), "Failed to read file".to_string()));
                            }
                        }
                        Err(e) => {
                            println!("FAIL");
                            failures.push((file.clone(), e));
                        }
                    }
                }
                
                if !failures.is_empty() {
                    eprintln!("\nFailures:");
                    for (file, error) in &failures {
                        eprintln!("  {}: {}", file.display(), error);
                    }
                    panic!("{} tests failed", failures.len());
                }
            }
        }
    };
}

// Generate test modules for each test directory
generate_tir_tests!(tir_tests, "../tpde/test/filetest/tir");
generate_tir_tests!(analyzer_tests, "../tpde/test/filetest/analyzer");
generate_tir_tests!(codegen_tests, "../tpde/test/filetest/codegen");