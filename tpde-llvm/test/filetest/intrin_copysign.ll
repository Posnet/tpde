; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

define float @copysignf32(float %0, float %1) {
; X64-LABEL: <copysignf32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    andps xmm1, xmmword ptr <copysignf32+0x14>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    andps xmm0, xmmword ptr <copysignf32+0x1b>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    orps xmm0, xmm1
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <copysignf32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mvni v2.4s, #0x80, lsl #24
; ARM64-NEXT:    bif v0.16b, v1.16b, v2.16b
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %res = call float @llvm.copysign.f32(float %0, float %1)
  ret float %res
}

define double @copysignf64(double %0, double %1) {
; X64-LABEL: <copysignf64>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    andps xmm1, xmmword ptr <copysignf64+0x14>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    andps xmm0, xmmword ptr <copysignf64+0x1b>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    orps xmm0, xmm1
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <copysignf64>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    movi v2.16b, #0xff
; ARM64-NEXT:    fneg v2.2d, v2.2d
; ARM64-NEXT:    bif v0.16b, v1.16b, v2.16b
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %res = call double @llvm.copysign.f64(double %0, double %1)
  ret double %res
}

define float @copysignf32_noreuse(float %0, float %1) {
; X64-LABEL: <copysignf32_noreuse>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    andps xmm1, xmmword ptr <copysignf32_noreuse+0x14>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    movapd xmm2, xmm0
; X64-NEXT:    andps xmm2, xmmword ptr <copysignf32_noreuse+0x1f>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    orps xmm2, xmm1
; X64-NEXT:    addss xmm0, xmm2
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <copysignf32_noreuse>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mvni v2.4s, #0x80, lsl #24
; ARM64-NEXT:    mov v3.16b, v0.16b
; ARM64-NEXT:    bif v3.16b, v1.16b, v2.16b
; ARM64-NEXT:    fadd s0, s0, s3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cs = call float @llvm.copysign.f32(float %0, float %1)
  %res = fadd float %0, %cs
  ret float %res
}

define double @copysignf64_noreuse(double %0, double %1) {
; X64-LABEL: <copysignf64_noreuse>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    andps xmm1, xmmword ptr <copysignf64_noreuse+0x14>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    movapd xmm2, xmm0
; X64-NEXT:    andps xmm2, xmmword ptr <copysignf64_noreuse+0x1f>
; X64-NEXT:     R_X86_64_PC32 -0x4
; X64-NEXT:    orps xmm2, xmm1
; X64-NEXT:    addsd xmm0, xmm2
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <copysignf64_noreuse>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    movi v2.16b, #0xff
; ARM64-NEXT:    fneg v2.2d, v2.2d
; ARM64-NEXT:    mov v3.16b, v0.16b
; ARM64-NEXT:    bif v3.16b, v1.16b, v2.16b
; ARM64-NEXT:    fadd d0, d0, d3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cs = call double @llvm.copysign.f64(double %0, double %1)
  %res = fadd double %0, %cs
  ret double %res
}
