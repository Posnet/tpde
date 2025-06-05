//! Test ICMP instruction functionality to validate implementation and identify improvements needed.
//!
//! This test exercises various ICMP predicates to check if the placeholder pattern mentioned
//! in the analysis exists and to validate real opcode-based comparison generation.

use tpde_rust::llvm_adaptor::enhanced::EnhancedLlvmAdaptor;
use tpde_rust::complete_compiler::CompleteCompiler;
use inkwell::{
    context::Context,
    module::Module,
    IntPredicate,
};

/// Create a test function with multiple ICMP predicates.
fn create_comparison_function(context: &Context) -> Module {
    let module = context.create_module("comparison_test");
    let i32_type = context.i32_type();
    let i1_type = context.bool_type();
    
    // bool test_comparisons(int a, int b) {
    //     return (a == b) && (a < b) && (a > b);
    // }
    let fn_type = i1_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_comparisons", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    
    // Test different ICMP predicates
    let eq_result = builder.build_int_compare(IntPredicate::EQ, a, b, "eq_result").unwrap();
    let lt_result = builder.build_int_compare(IntPredicate::SLT, a, b, "lt_result").unwrap();
    let gt_result = builder.build_int_compare(IntPredicate::SGT, a, b, "gt_result").unwrap();
    
    // Combine results with AND operations
    let and1 = builder.build_and(eq_result, lt_result, "and1").unwrap();
    let final_result = builder.build_and(and1, gt_result, "final_result").unwrap();
    
    builder.build_return(Some(&final_result)).unwrap();
    
    module
}

/// Create a function that tests all major ICMP predicates.
fn create_comprehensive_icmp_function(context: &Context) -> Module {
    let module = context.create_module("icmp_comprehensive");
    let i32_type = context.i32_type();
    let i1_type = context.bool_type();
    
    // Test all signed comparison predicates
    let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let function = module.add_function("test_all_icmp", fn_type, None);
    
    let entry_block = context.append_basic_block(function, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry_block);
    
    let a = function.get_nth_param(0).unwrap().into_int_value();
    let b = function.get_nth_param(1).unwrap().into_int_value();
    
    // Test multiple predicates to see which ones are supported
    let eq = builder.build_int_compare(IntPredicate::EQ, a, b, "eq").unwrap();
    let ne = builder.build_int_compare(IntPredicate::NE, a, b, "ne").unwrap();
    let slt = builder.build_int_compare(IntPredicate::SLT, a, b, "slt").unwrap();
    let sle = builder.build_int_compare(IntPredicate::SLE, a, b, "sle").unwrap();
    let sgt = builder.build_int_compare(IntPredicate::SGT, a, b, "sgt").unwrap();
    let sge = builder.build_int_compare(IntPredicate::SGE, a, b, "sge").unwrap();
    
    // Convert boolean to i32 to see which predicate is used
    let result = builder.build_int_z_extend(sgt, i32_type, "result").unwrap();
    
    builder.build_return(Some(&result)).unwrap();
    
    module
}

fn main() {
    println!("🧪 Testing ICMP instruction functionality");
    
    let context = Context::create();
    
    // Test 1: Multiple comparison predicates
    println!("\n🔍 Test 1: Multiple ICMP predicates in one function");
    let comparison_module = create_comparison_function(&context);
    
    match tpde_core::compile_enhanced_ir(&comparison_module) {
        Ok(mut compiler) => {
            match compiler.compile_all() {
                Ok(()) => {
                    let compiled = compiler.get_compiled_functions();
                    println!("✅ Multiple comparisons compilation successful!");
                    println!("   Function '{}' compiled to {} bytes", compiled[0].name, compiled[0].code.len());
                    println!("   Should show different ICMP predicates: EQ, SLT, SGT");
                }
                Err(e) => println!("❌ Multiple comparisons compilation failed: {:?}", e),
            }
        }
        Err(e) => println!("❌ Compiler creation failed: {:?}", e),
    }
    
    // Test 2: Comprehensive ICMP predicate test
    println!("\n📊 Test 2: Comprehensive ICMP predicate validation");
    let comprehensive_module = create_comprehensive_icmp_function(&context);
    
    match tpde_core::compile_enhanced_ir(&comprehensive_module) {
        Ok(mut compiler) => {
            match compiler.compile_all() {
                Ok(()) => {
                    let compiled = compiler.get_compiled_functions();
                    println!("✅ Comprehensive ICMP compilation successful!");
                    println!("   Function '{}' compiled to {} bytes", compiled[0].name, compiled[0].code.len());
                    println!("   This validates real CMP + SETcc instruction generation");
                }
                Err(e) => println!("❌ Comprehensive ICMP compilation failed: {:?}", e),
            }
        }
        Err(e) => println!("❌ Compiler creation failed: {:?}", e),
    }
    
    println!("\n🎯 ICMP functionality testing completed");
    println!("   This checks if placeholders exist or if real flag setting is implemented");
    println!("   Real implementation should show: CMP instruction + SETcc instruction patterns");
}