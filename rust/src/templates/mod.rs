// src/templates/mod.rs
// THIS FILE IS MANDATORY - IT DEFINES ALL INSTRUCTION TEMPLATES

#![allow(dead_code)]
#![cfg_attr(feature = "template-extraction", no_std)]

use tpde_macros::template_function;

// MANDATORY: Support ALL x86_64 microarchitecture levels (v1, v2, v3, v4)
// Each template MUST be annotated with the minimum required level

// --------------------------
// loads - x86_64-v1 (Baseline SSE, SSE2)
// --------------------------

#[template_function(target = "x86_64-v1")]
pub unsafe fn load_i8(ptr: *const i8) -> i8 { *ptr }

#[template_function(target = "x86_64-v1")]
pub unsafe fn load_i16(ptr: *const i16) -> i16 { *ptr }

#[template_function(target = "x86_64-v1")]
pub unsafe fn load_i32(ptr: *const i32) -> i32 { *ptr }

#[template_function(target = "x86_64-v1")]
pub unsafe fn load_i64(ptr: *const i64) -> i64 { *ptr }

#[template_function(target = "x86_64-v1")]
pub unsafe fn load_f32(ptr: *const f32) -> f32 { *ptr }

#[template_function(target = "x86_64-v1")]
pub unsafe fn load_f64(ptr: *const f64) -> f64 { *ptr }

// --------------------------
// stores - x86_64-v1
// --------------------------

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_i8(ptr: *mut i8, value: i8) { *ptr = value; }

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_i16(ptr: *mut i16, value: i16) { *ptr = value; }

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_i32(ptr: *mut i32, value: i32) { *ptr = value; }

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_i64(ptr: *mut i64, value: i64) { *ptr = value; }

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_f32(ptr: *mut f32, value: f32) { *ptr = value; }

#[template_function(target = "x86_64-v1")]
pub unsafe fn store_f64(ptr: *mut f64, value: f64) { *ptr = value; }

// --------------------------
// integer arithmetic - x86_64-v1
// --------------------------

#[template_function(target = "x86_64-v1")]
pub fn add_i32(a: i32, b: i32) -> i32 { a.wrapping_add(b) }

#[template_function(target = "x86_64-v1")]
pub fn add_i64(a: i64, b: i64) -> i64 { a.wrapping_add(b) }

#[template_function(target = "x86_64-v1")]
pub fn sub_i32(a: i32, b: i32) -> i32 { a.wrapping_sub(b) }

#[template_function(target = "x86_64-v1")]
pub fn sub_i64(a: i64, b: i64) -> i64 { a.wrapping_sub(b) }

#[template_function(target = "x86_64-v1")]
pub fn mul_i32(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }

#[template_function(target = "x86_64-v1")]
pub fn mul_i64(a: i64, b: i64) -> i64 { a.wrapping_mul(b) }

#[template_function(target = "x86_64-v1")]
pub fn udiv_i32(a: u32, b: u32) -> u32 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn udiv_i64(a: u64, b: u64) -> u64 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn sdiv_i32(a: i32, b: i32) -> i32 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn sdiv_i64(a: i64, b: i64) -> i64 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn urem_i32(a: u32, b: u32) -> u32 { a % b }

#[template_function(target = "x86_64-v1")]
pub fn urem_i64(a: u64, b: u64) -> u64 { a % b }

#[template_function(target = "x86_64-v1")]
pub fn srem_i32(a: i32, b: i32) -> i32 { a % b }

#[template_function(target = "x86_64-v1")]
pub fn srem_i64(a: i64, b: i64) -> i64 { a % b }

#[template_function(target = "x86_64-v1")]
pub fn and_i32(a: i32, b: i32) -> i32 { a & b }

#[template_function(target = "x86_64-v1")]
pub fn and_i64(a: i64, b: i64) -> i64 { a & b }

#[template_function(target = "x86_64-v1")]
pub fn or_i32(a: i32, b: i32) -> i32 { a | b }

#[template_function(target = "x86_64-v1")]
pub fn or_i64(a: i64, b: i64) -> i64 { a | b }

#[template_function(target = "x86_64-v1")]
pub fn xor_i32(a: i32, b: i32) -> i32 { a ^ b }

#[template_function(target = "x86_64-v1")]
pub fn xor_i64(a: i64, b: i64) -> i64 { a ^ b }

#[template_function(target = "x86_64-v1")]
pub fn shl_i32(a: i32, b: i32) -> i32 { a << (b & 31) }

#[template_function(target = "x86_64-v1")]
pub fn shl_i64(a: i64, b: i64) -> i64 { a << (b & 63) }

#[template_function(target = "x86_64-v1")]
pub fn lshr_i32(a: u32, b: u32) -> u32 { a >> (b & 31) }

#[template_function(target = "x86_64-v1")]
pub fn lshr_i64(a: u64, b: u64) -> u64 { a >> (b & 63) }

#[template_function(target = "x86_64-v1")]
pub fn ashr_i32(a: i32, b: i32) -> i32 { a >> (b & 31) }

#[template_function(target = "x86_64-v1")]
pub fn ashr_i64(a: i64, b: i64) -> i64 { a >> (b & 63) }

// --------------------------
// float arithmetic - x86_64-v1 (SSE/SSE2)
// --------------------------

#[template_function(target = "x86_64-v1")]
pub fn add_f32(a: f32, b: f32) -> f32 { a + b }

#[template_function(target = "x86_64-v1")]
pub fn add_f64(a: f64, b: f64) -> f64 { a + b }

#[template_function(target = "x86_64-v1")]
pub fn sub_f32(a: f32, b: f32) -> f32 { a - b }

#[template_function(target = "x86_64-v1")]
pub fn sub_f64(a: f64, b: f64) -> f64 { a - b }

#[template_function(target = "x86_64-v1")]
pub fn mul_f32(a: f32, b: f32) -> f32 { a * b }

#[template_function(target = "x86_64-v1")]
pub fn mul_f64(a: f64, b: f64) -> f64 { a * b }

#[template_function(target = "x86_64-v1")]
pub fn div_f32(a: f32, b: f32) -> f32 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn div_f64(a: f64, b: f64) -> f64 { a / b }

#[template_function(target = "x86_64-v1")]
pub fn fneg_f32(a: f32) -> f32 { -a }

#[template_function(target = "x86_64-v1")]
pub fn fneg_f64(a: f64) -> f64 { -a }

#[template_function(target = "x86_64-v1")]
pub fn fabs_f32(a: f32) -> f32 { a.abs() }

#[template_function(target = "x86_64-v1")]
pub fn fabs_f64(a: f64) -> f64 { a.abs() }

#[template_function(target = "x86_64-v1")]
pub fn sqrt_f32(a: f32) -> f32 { a.sqrt() }

#[template_function(target = "x86_64-v1")]
pub fn sqrt_f64(a: f64) -> f64 { a.sqrt() }

#[template_function(target = "x86_64-v1")]
pub fn fmin_f32(a: f32, b: f32) -> f32 { a.min(b) }

#[template_function(target = "x86_64-v1")]
pub fn fmin_f64(a: f64, b: f64) -> f64 { a.min(b) }

#[template_function(target = "x86_64-v1")]
pub fn fmax_f32(a: f32, b: f32) -> f32 { a.max(b) }

#[template_function(target = "x86_64-v1")]
pub fn fmax_f64(a: f64, b: f64) -> f64 { a.max(b) }

// --------------------------
// float conversions - x86_64-v1
// --------------------------

#[template_function(target = "x86_64-v1")]
pub fn f64_to_f32(a: f64) -> f32 { a as f32 }

#[template_function(target = "x86_64-v1")]
pub fn f32_to_f64(a: f32) -> f64 { a as f64 }

#[template_function(target = "x86_64-v1")]
pub fn f32_to_i32(a: f32) -> i32 { a as i32 }

#[template_function(target = "x86_64-v1")]
pub fn f32_to_u32(a: f32) -> u32 { a as u32 }

#[template_function(target = "x86_64-v1")]
pub fn f32_to_i64(a: f32) -> i64 { a as i64 }

#[template_function(target = "x86_64-v1")]
pub fn f32_to_u64(a: f32) -> u64 { a as u64 }

#[template_function(target = "x86_64-v1")]
pub fn f64_to_i32(a: f64) -> i32 { a as i32 }

#[template_function(target = "x86_64-v1")]
pub fn f64_to_u32(a: f64) -> u32 { a as u32 }

#[template_function(target = "x86_64-v1")]
pub fn f64_to_i64(a: f64) -> i64 { a as i64 }

#[template_function(target = "x86_64-v1")]
pub fn f64_to_u64(a: f64) -> u64 { a as u64 }

#[template_function(target = "x86_64-v1")]
pub fn i32_to_f32(a: i32) -> f32 { a as f32 }

#[template_function(target = "x86_64-v1")]
pub fn u32_to_f32(a: u32) -> f32 { a as f32 }

#[template_function(target = "x86_64-v1")]
pub fn i64_to_f32(a: i64) -> f32 { a as f32 }

#[template_function(target = "x86_64-v1")]
pub fn u64_to_f32(a: u64) -> f32 { a as f32 }

#[template_function(target = "x86_64-v1")]
pub fn i32_to_f64(a: i32) -> f64 { a as f64 }

#[template_function(target = "x86_64-v1")]
pub fn u32_to_f64(a: u32) -> f64 { a as f64 }

#[template_function(target = "x86_64-v1")]
pub fn i64_to_f64(a: i64) -> f64 { a as f64 }

#[template_function(target = "x86_64-v1")]
pub fn u64_to_f64(a: u64) -> f64 { a as f64 }

// --------------------------
// comparisons - x86_64-v1
// --------------------------

#[template_function(target = "x86_64-v1")]
pub fn icmp_eq_i32(a: i32, b: i32) -> bool { a == b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ne_i32(a: i32, b: i32) -> bool { a != b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_slt_i32(a: i32, b: i32) -> bool { a < b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_sle_i32(a: i32, b: i32) -> bool { a <= b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_sgt_i32(a: i32, b: i32) -> bool { a > b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_sge_i32(a: i32, b: i32) -> bool { a >= b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ult_i32(a: u32, b: u32) -> bool { a < b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ule_i32(a: u32, b: u32) -> bool { a <= b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ugt_i32(a: u32, b: u32) -> bool { a > b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_uge_i32(a: u32, b: u32) -> bool { a >= b }

// Same for i64
#[template_function(target = "x86_64-v1")]
pub fn icmp_eq_i64(a: i64, b: i64) -> bool { a == b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ne_i64(a: i64, b: i64) -> bool { a != b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_slt_i64(a: i64, b: i64) -> bool { a < b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_sle_i64(a: i64, b: i64) -> bool { a <= b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_sgt_i64(a: i64, b: i64) -> bool { a > b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_sge_i64(a: i64, b: i64) -> bool { a >= b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ult_i64(a: u64, b: u64) -> bool { a < b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ule_i64(a: u64, b: u64) -> bool { a <= b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_ugt_i64(a: u64, b: u64) -> bool { a > b }

#[template_function(target = "x86_64-v1")]
pub fn icmp_uge_i64(a: u64, b: u64) -> bool { a >= b }

// Float comparisons
#[template_function(target = "x86_64-v1")]
pub fn fcmp_oeq_f32(a: f32, b: f32) -> bool { a == b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_one_f32(a: f32, b: f32) -> bool { a != b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_olt_f32(a: f32, b: f32) -> bool { a < b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_ole_f32(a: f32, b: f32) -> bool { a <= b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_ogt_f32(a: f32, b: f32) -> bool { a > b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_oge_f32(a: f32, b: f32) -> bool { a >= b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_oeq_f64(a: f64, b: f64) -> bool { a == b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_one_f64(a: f64, b: f64) -> bool { a != b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_olt_f64(a: f64, b: f64) -> bool { a < b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_ole_f64(a: f64, b: f64) -> bool { a <= b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_ogt_f64(a: f64, b: f64) -> bool { a > b }

#[template_function(target = "x86_64-v1")]
pub fn fcmp_oge_f64(a: f64, b: f64) -> bool { a >= b }

// --------------------------
// select (ternary) - x86_64-v1
// --------------------------

#[template_function(target = "x86_64-v1", inline_mode = "always")]
pub fn select_i32(cond: bool, a: i32, b: i32) -> i32 {
    if cond { a } else { b }
}

#[template_function(target = "x86_64-v1", inline_mode = "always")]
pub fn select_i64(cond: bool, a: i64, b: i64) -> i64 {
    if cond { a } else { b }
}

#[template_function(target = "x86_64-v1", inline_mode = "always")]
pub fn select_f32(cond: bool, a: f32, b: f32) -> f32 {
    if cond { a } else { b }
}

#[template_function(target = "x86_64-v1", inline_mode = "always")]
pub fn select_f64(cond: bool, a: f64, b: f64) -> f64 {
    if cond { a } else { b }
}

// --------------------------
// x86_64-v2: SSSE3, SSE4.1, SSE4.2, POPCNT
// --------------------------

#[template_function(target = "x86_64-v2")]
pub fn popcnt_i32(a: i32) -> i32 {
    a.count_ones() as i32
}

#[template_function(target = "x86_64-v2")]
pub fn popcnt_i64(a: i64) -> i32 {
    a.count_ones() as i32
}

// --------------------------
// x86_64-v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE
// --------------------------

#[template_function(target = "x86_64-v3")]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    // Rust will optimize this to FMA instruction with target-cpu=x86-64-v3
    a.mul_add(b, c)
}

#[template_function(target = "x86_64-v3")]
pub fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[template_function(target = "x86_64-v3")]
pub fn lzcnt_i32(a: i32) -> i32 {
    a.leading_zeros() as i32
}

#[template_function(target = "x86_64-v3")]
pub fn lzcnt_i64(a: i64) -> i32 {
    a.leading_zeros() as i32
}

#[template_function(target = "x86_64-v3")]
pub fn tzcnt_i32(a: i32) -> i32 {
    a.trailing_zeros() as i32
}

#[template_function(target = "x86_64-v3")]
pub fn tzcnt_i64(a: i64) -> i32 {
    a.trailing_zeros() as i32
}

// Bit manipulation instructions (BMI1/BMI2)
#[template_function(target = "x86_64-v3")]
pub fn andn_i32(a: i32, b: i32) -> i32 {
    !a & b
}

#[template_function(target = "x86_64-v3")]
pub fn andn_i64(a: i64, b: i64) -> i64 {
    !a & b
}

// TODO: Add more x86_64-v3 operations (AVX, AVX2 vector operations)

// --------------------------
// x86_64-v4: AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL
// --------------------------

// TODO: Add AVX512 operations for v4

// Note: This is a partial implementation. The full implementation should include:
// - All operations from encode_template.c
// - Vector operations for each microarchitecture level
// - Additional intrinsics and special operations
// - Overflow operations with carry flags
// - Rotate operations
// - Byte swap operations
// - More conversion operations