//! Integration tests for TIR (Test IR) files.
//!
//! These tests replicate the C++ filetest/tir tests in a Rust-native way,
//! using the #[test] framework instead of FileCheck.

use std::fs;
use std::path::Path;
use tpde::test_ir::TestIR;

/// Helper to load and parse a TIR file from the test directory
fn load_tir_file(filename: &str) -> TestIR {
    let path = Path::new("../tpde/test/filetest/tir").join(filename);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));

    TestIR::parse(&contents).unwrap_or_else(|e| panic!("Failed to parse {filename}: {e}"))
}

/// Helper to check if output contains expected patterns
fn check_output_contains(output: &str, patterns: &[&str]) {
    for pattern in patterns {
        assert!(
            output.contains(pattern),
            "Output missing expected pattern: '{pattern}'\nFull output:\n{output}"
        );
    }
}

#[test]
fn test_add_tir() {
    let ir = load_tir_file("add.tir");
    let output = ir.print();

    check_output_contains(
        &output,
        &[
            "Printing IR",
            "Function func",
            "Block entry",
            "Value a (any)",
            "Value b (any)",
            "Value c (add)",
            "Op a",
            "Op b",
            "Value (terminate)",
        ],
    );

    // Verify structure
    assert_eq!(ir.functions.len(), 1);
    assert_eq!(ir.functions[0].name, "func");
    assert_eq!(ir.blocks.len(), 1);
    assert_eq!(ir.values.len(), 4); // a, b, c, terminate
}

#[test]
fn test_br_tir() {
    let ir = load_tir_file("br.tir");
    let output = ir.print();

    // Test br1 function
    check_output_contains(
        &output,
        &[
            "Function br1",
            "Block entry",
            "Succ secBlock",
            "Value (br)",
            "Op ^secBlock",
            "Block secBlock",
            "Succ retBlock",
            "Op ^retBlock",
            "Block retBlock",
            "Value (terminate)",
        ],
    );

    // Test condbr1 function
    check_output_contains(
        &output,
        &[
            "Function condbr1",
            "Block entry",
            "Succ ret1",
            "Succ ret2",
            "Value val",
            "Value (condbr)",
            "Op val",
            "Op ^ret1",
            "Op ^ret2",
            "Block ret1",
            "Block ret2",
        ],
    );

    // Verify structure
    assert_eq!(ir.functions.len(), 2);
    assert_eq!(ir.functions[0].name, "br1");
    assert_eq!(ir.functions[1].name, "condbr1");
}

#[test]
fn test_simple_tir() {
    let ir = load_tir_file("simple.tir");
    let output = ir.print();

    // Test myfunc
    check_output_contains(
        &output,
        &[
            "Function myfunc",
            "Argument a",
            "Block entry",
            "Succ ret",
            "Succ other",
            "Value alloca (alloca)",
            "Op $16",
            "Op $8",
            "Value val",
            "Value dep",
            "Op alloca",
            "Op val",
            "Op tbd",
            "Value tbd",
            "Op val",
            "Value (jump)",
            "Op ^ret",
            "Op ^other",
            "Block other",
            "Succ ret",
            "Value val2",
            "Op tbd",
            "Value (jump)",
            "Op ^ret",
            "Block ret",
            "PHI phi",
            "val from entry",
            "val2 from other",
            "Value (terminate)",
        ],
    );

    // Test secfunc
    check_output_contains(
        &output,
        &[
            "Function secfunc",
            "Argument a",
            "Argument b",
            "Block entry",
            "Value val",
            "Value (ret)",
            "Op val",
        ],
    );

    // Verify PHI node structure
    let phi_values: Vec<_> = ir
        .values
        .iter()
        .filter(|v| v.value_type == tpde::test_ir::ValueType::Phi)
        .collect();
    assert_eq!(phi_values.len(), 1);
    assert_eq!(phi_values[0].name, "phi");
    assert_eq!(phi_values[0].op_count, 2); // Two incoming values
}

#[test]
fn test_alloca_align_tir() {
    let ir = load_tir_file("alloca_align.tir");
    let output = ir.print();

    check_output_contains(
        &output,
        &[
            "Function test",
            "Block entry",
            "Value a1 (alloca)",
            "Op $8",
            "Op $1",
            "Value a2 (alloca)",
            "Op $8",
            "Op $2",
            "Value a4 (alloca)",
            "Op $8",
            "Op $4",
            "Value a8 (alloca)",
            "Op $8",
            "Op $8",
            "Value a16 (alloca)",
            "Op $8",
            "Op $16",
            "Value a32 (alloca)",
            "Op $8",
            "Op $32",
            "Value (terminate)",
        ],
    );

    // Verify alloca instructions
    let allocas: Vec<_> = ir
        .values
        .iter()
        .filter(|v| v.op == tpde::test_ir::Operation::Alloca)
        .collect();
    assert_eq!(allocas.len(), 6);

    // Check alignment values
    let expected_aligns = [1, 2, 4, 8, 16, 32];
    for (i, alloca) in allocas.iter().enumerate() {
        let align_idx = (alloca.op_begin_idx + 1) as usize;
        let align = ir.value_operands[align_idx];
        assert_eq!(
            align, expected_aligns[i],
            "Alignment mismatch for {}",
            alloca.name
        );
    }
}

#[test]
fn test_call_tir() {
    let ir = load_tir_file("call.tir");
    let output = ir.print();

    check_output_contains(
        &output,
        &[
            "Function caller",
            "Argument a",
            "Argument b",
            "Block entry",
            "Value c (call)",
            "Op a",
            "Op b",
            "Value (ret)",
            "Op c",
        ],
    );

    check_output_contains(
        &output,
        &[
            "Function callee",
            "Argument x",
            "Argument y",
            "Block entry",
            "Value z (add)",
            "Op x",
            "Op y",
            "Value (ret)",
            "Op z",
        ],
    );

    // Verify call instruction
    let calls: Vec<_> = ir
        .values
        .iter()
        .filter(|v| v.op == tpde::test_ir::Operation::Call)
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "c");
    assert_eq!(calls[0].op_count, 2); // Two arguments
}

#[test]
fn test_func_tir() {
    let ir = load_tir_file("func.tir");
    let output = ir.print();

    check_output_contains(
        &output,
        &[
            "Printing IR",
            "Function empty_func",
            "Block entry",
            "Value (terminate)",
        ],
    );

    // Verify minimal function structure
    assert_eq!(ir.functions.len(), 1);
    assert_eq!(ir.functions[0].name, "empty_func");
    assert_eq!(ir.blocks.len(), 1);
    assert_eq!(ir.blocks[0].name, "entry");
}

#[test]
fn test_duplicate_funcs_tir() {
    let ir = load_tir_file("duplicate_funcs.tir");
    let output = ir.print();

    check_output_contains(&output, &["Function func1", "Function func2"]);

    // Verify we have two functions
    assert_eq!(ir.functions.len(), 2);
    assert_eq!(ir.functions[0].name, "func1");
    assert_eq!(ir.functions[1].name, "func2");
}

// Additional verification tests for parser edge cases

#[test]
fn test_parser_comments() {
    let tir = r#"
; This is a comment
func() { ; Another comment
  entry: ; Comment after label
    ; Comment line
    %val = ; Comment after =
    terminate ; Final comment
}
"#;

    let ir = TestIR::parse(tir).unwrap();
    assert_eq!(ir.functions.len(), 1);
    assert_eq!(ir.values.len(), 2); // val and terminate
}

#[test]
fn test_parser_force_fixed_assignment() {
    let tir = r#"
func() {
  entry:
    %normal =
    %fixed! =
    terminate
}
"#;

    let ir = TestIR::parse(tir).unwrap();
    assert!(!ir.values[0].force_fixed_assignment);
    assert!(ir.values[1].force_fixed_assignment);
}

#[test]
fn test_parser_error_undefined_value() {
    let tir = r#"
func() {
  entry:
    %c = add %a, %b
    terminate
}
"#;

    let result = TestIR::parse(tir);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Undefined value reference"));
}

#[test]
fn test_parser_error_undefined_block() {
    let tir = r#"
func() {
  entry:
    br ^undefined
}
"#;

    let result = TestIR::parse(tir);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Undefined block reference"));
}

#[test]
fn test_parser_error_invalid_phi_location() {
    let tir = r#"
func() {
  entry:
    %val =
    %phi = phi [^entry, %val]
    terminate
}
"#;

    let result = TestIR::parse(tir);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("PHI nodes must be at the beginning"));
}
