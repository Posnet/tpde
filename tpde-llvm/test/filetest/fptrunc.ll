; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

define float @fptrunc_f64tof32(double %0) {
; X64-LABEL: <fptrunc_f64tof32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    cvtsd2ss xmm0, xmm0
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fptrunc_f64tof32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    fcvt s0, d0
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = fptrunc double %0 to float
  ret float %1
}

define float @fptrunc_f128tof32(fp128 %in) {
; X64-LABEL: <fptrunc_f128tof32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __trunctfsf2-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fptrunc_f128tof32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x70 <fptrunc_f128tof32+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __trunctfsf2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %trunc = fptrunc fp128 %in to float
  ret float %trunc
}

define double @fptrunc_f128tof64(fp128 %in) {
; X64-LABEL: <fptrunc_f128tof64>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __trunctfdf2-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fptrunc_f128tof64>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0xb0 <fptrunc_f128tof64+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __trunctfdf2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %trunc = fptrunc fp128 %in to double
  ret double %trunc
}

