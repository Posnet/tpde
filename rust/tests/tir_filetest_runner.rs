//! Dynamic test runner for TIR filetest files.
//!
//! This module discovers and runs all .tir files from the C++ test directories,
//! executing them with the same capabilities as the C++ tpde_test harness.

use std::fs;
use std::path::{Path, PathBuf};
use tpde::core::{Analyzer, IrAdaptor};
use tpde::test_ir::{TestIR, TestIRAdaptor};

#[derive(Default)]
struct TestConfig {
    print_ir: bool,
    print_rpo: bool,
    print_liveness: bool,
    print_loops: bool,
    print_layout: bool,
    run_until: String,
    no_fixed_assignments: bool,
    obj_out_path: Option<String>,
    is_arm64: bool,
    expect_failure: bool,
}

impl TestConfig {
    fn default() -> Self {
        Self {
            run_until: "codegen".to_string(),
            ..Default::default()
        }
    }

    fn parse_run_line(&mut self, run_line: &str) {
        // Check if the command is expected to fail
        let run_line = if let Some(cmd) = run_line.trim().strip_prefix("not ") {
            self.expect_failure = true;
            cmd
        } else {
            run_line
        };

        // Parse flags
        self.is_arm64 = run_line.contains("--arch=a64") || run_line.contains("--arch=arm64");
        self.print_ir = run_line.contains("--print-ir");
        self.print_rpo = run_line.contains("--print-rpo");
        self.print_liveness = run_line.contains("--print-liveness");
        self.print_loops = run_line.contains("--print-loops");
        self.print_layout = run_line.contains("--print-layout");
        self.no_fixed_assignments = run_line.contains("--no-fixed-assignments");

        // Parse --run-until=
        if let Some(pos) = run_line.find("--run-until=") {
            let start = pos + "--run-until=".len();
            let end = run_line[start..]
                .find(' ')
                .unwrap_or(run_line.len() - start);
            self.run_until = run_line[start..start + end].to_string();
        }

        // Parse -o output.o
        if let Some(pos) = run_line.find("-o ") {
            let start = pos + 3;
            let end = run_line[start..]
                .find(' ')
                .unwrap_or(run_line.len() - start);
            self.obj_out_path = Some(run_line[start..start + end].to_string());
        }
    }
}

/// Discovers all .tir files in a directory recursively
fn discover_tir_files(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .flat_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                discover_tir_files(&path)
            } else if path.extension().and_then(|s| s.to_str()) == Some("tir") {
                vec![path]
            } else {
                vec![]
            }
        })
        .collect()
}

/// Run a single TIR file and return its output
fn run_tir_file(path: &Path) -> Result<String, String> {
    // Initialize logging if not already done
    let _ = env_logger::builder().is_test(true).try_init();

    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    // Parse RUN directive
    let mut config = TestConfig::default();

    for line in content.lines() {
        if let Some(run_line) = line.strip_prefix("; RUN:") {
            config.parse_run_line(run_line);
        }
    }

    // Skip ARM64-specific tests for now since we only support x86-64
    if config.is_arm64 {
        return Ok("Skipping ARM64-specific test".to_string());
    }

    // Parse the TIR content
    let parse_result = TestIR::parse(&content);

    // Handle expected failures
    if config.expect_failure {
        match parse_result {
            Err(e) => return Ok(format!("Expected failure: {e}")),
            Ok(_) => return Err("Expected parse failure but succeeded".to_string()),
        }
    }

    let ir = parse_result?;

    // Check if IR contains ARM64-specific instructions (like tbz)
    for value in &ir.values {
        if value.op == tpde::test_ir::Operation::Tbz {
            return Ok("Skipping ARM64-specific test (contains tbz instruction)".to_string());
        }
    }
    let mut output = Vec::new();

    // Print IR if requested
    if config.print_ir {
        output.push("Printing IR".to_string());
        output.push(ir.to_string());
    }

    // Run analyzer if needed
    if config.print_rpo
        || config.print_liveness
        || config.print_loops
        || config.print_layout
        || config.run_until == "analyzer"
    {
        let mut adaptor = TestIRAdaptor::new(&ir);
        let mut analyzer = Analyzer::new();

        // Process each function
        let funcs: Vec<_> = adaptor.funcs().collect();
        for func in funcs {
            let func_name = adaptor.func_link_name(func).to_string();
            adaptor.switch_func(func);
            analyzer.switch_func(&mut adaptor, func);

            if config.print_rpo {
                output.push(format!("RPO for func {func_name}"));
                let rpo = analyzer.order();
                for (idx, &block) in rpo.iter().enumerate() {
                    let block_name = adaptor.block_name(block);
                    output.push(format!("{idx}: {block_name}"));
                }
                output.push("End RPO".to_string());
            }

            if config.print_layout {
                output.push(format!("Block Layout for {func_name}"));
                let layout = analyzer.block_layout();
                for (idx, &block) in layout.iter().enumerate() {
                    let block_name = adaptor.block_name(block);
                    output.push(format!("{idx}: {block_name}"));
                }
                output.push("End Block Layout".to_string());
            }

            if config.print_loops {
                output.push(format!("Loops for {func_name}"));
                let loops = analyzer.loops();
                for (idx, loop_info) in loops.iter().enumerate() {
                    output.push(format!(
                        "{}: level {}, parent {}, {}->{}",
                        idx, loop_info.level, loop_info.parent, loop_info.begin, loop_info.end
                    ));
                }
                output.push("End Loops".to_string());
            }

            if config.print_liveness {
                output.push(format!("Liveness for {func_name}"));

                // Get block names for mapping - use block layout instead of RPO
                let layout = analyzer.block_layout();
                let block_names: Vec<String> = layout
                    .iter()
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
                    let should_ignore = val_info.op == tpde::test_ir::Operation::Alloca
                        || (val_info.value_type == tpde::test_ir::ValueType::Terminator
                            && !val_info.op.info().is_def);

                    if should_ignore {
                        output.push(format!("{local_idx}: ignored"));
                    } else if let Some(liveness) = analyzer.liveness(local_idx) {
                        let first_block = block_names
                            .get(liveness.first)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        let last_block = block_names
                            .get(liveness.last)
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

    // Handle codegen stage
    if config.run_until == "codegen" || config.obj_out_path.is_some() {
        // Actually compile to object code
        use bumpalo::Bump;
        use tpde::core::session::CompilationSession;
        use tpde::test_ir::TestIRCompiler;

        let arena = Bump::new();
        let session = CompilationSession::new(&arena);

        let compiler = TestIRCompiler::new(&ir, &session, config.no_fixed_assignments)
            .map_err(|e| format!("Failed to create compiler: {e:?}"))?;

        let object_code = compiler
            .compile()
            .map_err(|e| format!("Compilation failed: {e:?}"))?;

        // Write object file if path provided
        if let Some(out_path) = &config.obj_out_path {
            // Handle %t placeholder
            let actual_path = if out_path.contains("%t") {
                // Create temp directory
                let temp_dir =
                    std::env::temp_dir().join(format!("tpde_test_{}", std::process::id()));
                std::fs::create_dir_all(&temp_dir).ok();
                out_path.replace("%t", temp_dir.to_str().unwrap())
            } else {
                out_path.clone()
            };

            // Ensure parent directory exists
            if let Some(parent) = std::path::Path::new(&actual_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }

            std::fs::write(&actual_path, &object_code)
                .map_err(|e| format!("Failed to write object file: {e}"))?;

            // For codegen tests, we need to disassemble the object
            if config.run_until == "codegen" && actual_path.ends_with(".o") {
                // Run objdump on the generated object file
                // Note: macOS objdump doesn't support -Mintel-syntax, so we'll use a different approach
                let objdump_output = if cfg!(target_os = "macos") {
                    // Try llvm-objdump first, fall back to objdump
                    match std::process::Command::new("llvm-objdump")
                        .arg("-d")
                        .arg("-x86-asm-syntax=intel")
                        .arg("--no-show-raw-insn")
                        .arg(&actual_path)
                        .output()
                    {
                        Ok(output) => output,
                        Err(_) => {
                            // Fall back to regular objdump without Intel syntax
                            std::process::Command::new("objdump")
                                .arg("-d")
                                .arg(&actual_path)
                                .output()
                                .map_err(|e| format!("Failed to run objdump: {e}"))?
                        }
                    }
                } else {
                    std::process::Command::new("objdump")
                        .arg("-Mintel-syntax")
                        .arg("--no-addresses")
                        .arg("--no-show-raw-insn")
                        .arg("--disassemble")
                        .arg(&actual_path)
                        .output()
                        .map_err(|e| format!("Failed to run objdump: {e}"))?
                };

                if objdump_output.status.success() {
                    let disasm = String::from_utf8_lossy(&objdump_output.stdout);
                    output.push(disasm.to_string());
                } else {
                    let stderr = String::from_utf8_lossy(&objdump_output.stderr);
                    return Err(format!("objdump failed: {stderr}"));
                }
            }
        } else {
            output.push(format!(
                "Generated {} bytes of object code",
                object_code.len()
            ));
        }
    }

    Ok(output.join("\n"))
}

/// Validate output against CHECK directives
fn validate_output(output: &str, content: &str) -> Result<(), String> {
    // Special case for skipped tests
    if output.starts_with("Skipping ARM64-specific test") {
        return Ok(());
    }

    // Special case for expected failures
    if output.starts_with("Expected failure:") {
        return Ok(());
    }

    let mut checks = Vec::new();

    // Extract CHECK directives
    for (current_line, line) in content.lines().enumerate() {
        if let Some(check) = line.strip_prefix("; CHECK:") {
            checks.push((current_line, "CHECK", check.trim()));
        } else if let Some(check) = line.strip_prefix("; CHECK-NEXT:") {
            checks.push((current_line, "CHECK-NEXT", check.trim()));
        } else if let Some(check) = line.strip_prefix("; CHECK-LABEL:") {
            checks.push((current_line, "CHECK-LABEL", check.trim()));
        } else if line.starts_with("; CHECK-EMPTY") {
            checks.push((current_line, "CHECK-EMPTY", ""));
        }
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
