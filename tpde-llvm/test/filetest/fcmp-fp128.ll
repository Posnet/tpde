; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

define i1 @fcmp_f128_false(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_false>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, 0x0
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_false>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov w0, #0x0 // =0
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp false fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_true(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_true>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, 0x1
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_true>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x0, #0x1 // =1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp true fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_oge(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_oge>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __getf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setge al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_oge>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0xb0 <fcmp_f128_oge+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __getf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, ge
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp oge fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_ord(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_ord>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __unordtf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    sete al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_ord>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x100 <fcmp_f128_ord+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __unordtf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, eq
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp ord fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_oeq(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_oeq>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __eqtf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    sete al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_oeq>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x150 <fcmp_f128_oeq+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __eqtf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, eq
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp oeq fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_ogt(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_ogt>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __gttf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setg al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_ogt>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x1a0 <fcmp_f128_ogt+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __gttf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, gt
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp ogt fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_olt(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_olt>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __lttf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setl al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_olt>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x1f0 <fcmp_f128_olt+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __lttf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, lt
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp olt fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_ole(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_ole>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __letf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setle al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_ole>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x240 <fcmp_f128_ole+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __letf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, le
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp ole fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_uno(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_uno>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __unordtf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setne al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_uno>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x290 <fcmp_f128_uno+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __unordtf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, ne
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp uno fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_ugt(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_ugt>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __letf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setg al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_ugt>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x2e0 <fcmp_f128_ugt+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __letf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, gt
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp ugt fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_uge(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_uge>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __lttf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setge al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_uge>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x330 <fcmp_f128_uge+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __lttf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, ge
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp uge fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_ult(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_ult>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __getf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setl al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_ult>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x380 <fcmp_f128_ult+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __getf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, lt
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp ult fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_ule(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_ule>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __gttf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setle al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_ule>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x3d0 <fcmp_f128_ule+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __gttf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, le
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp ule fp128 %0, %1
  ret i1 %cmp
}

define i1 @fcmp_f128_une(fp128 %0, fp128 %1) {
; X64-LABEL: <fcmp_f128_une>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __netf2-0x4
; X64-NEXT:    test rax, rax
; X64-NEXT:    setne al
; X64-NEXT:    movzx eax, al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <fcmp_f128_une>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x420 <fcmp_f128_une+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __netf2
; ARM64-NEXT:    cmp w0, #0x0
; ARM64-NEXT:    cset w0, ne
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %cmp = fcmp une fp128 %0, %1
  ret i1 %cmp
}

; define i1 @fcmp_f128_one(fp128 %0, fp128 %1) {
;   %cmp = fcmp one fp128 %0, %1
;   ret i1 %cmp
; }

; define i1 @fcmp_f128_ueq(fp128 %0, fp128 %1) {
;   %cmp = fcmp ueq fp128 %0, %1
;   ret i1 %cmp
; }

; define i1 @fcmp_f128_one_nonan(fp128 %0, fp128 %1) {
;   %cmp = fcmp nnan one fp128 %0, %1
;   ret i1 %cmp
; }

; define i1 @fcmp_f128_oeq_0(fp128 %0) {
;   %cmp = fcmp oeq fp128 %0, 0xL00000000000000000000000000000000
;   ret i1 %cmp
; }
