; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

define void @lshr_i8_3(i8 %0) {
; X64-LABEL: <lshr_i8_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movzx edi, dil
; X64-NEXT:    shr edi, 0x3
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i8_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    uxtb w0, w0
; ARM64-NEXT:    lsr w0, w0, #3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i8 %0, 3
    ret void
}

define void @lshr_i8_i8(i8 %0, i8 %1) {
; X64-LABEL: <lshr_i8_i8>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movzx edi, dil
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr edi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i8_i8>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    uxtb w0, w0
; ARM64-NEXT:    lsr w0, w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i8 %0, %1
    ret void
}

define void @lshr_i16_3(i16 %0) {
; X64-LABEL: <lshr_i16_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movzx edi, di
; X64-NEXT:    shr edi, 0x3
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i16_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    uxth w0, w0
; ARM64-NEXT:    lsr w0, w0, #3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i16 %0, 3
    ret void
}

define void @lshr_i16_i16(i16 %0, i16 %1) {
; X64-LABEL: <lshr_i16_i16>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movzx edi, di
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr edi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i16_i16>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    uxth w0, w0
; ARM64-NEXT:    lsr w0, w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i16 %0, %1
    ret void
}

define void @lshr_i32_3(i32 %0) {
; X64-LABEL: <lshr_i32_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    shr edi, 0x3
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i32_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr w0, w0, #3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i32 %0, 3
    ret void
}

define void @lshr_i32_i32(i32 %0, i32 %1) {
; X64-LABEL: <lshr_i32_i32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr edi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i32_i32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr w0, w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i32 %0, %1
    ret void
}

define void @lshr_i64_3(i64 %0) {
; X64-LABEL: <lshr_i64_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    shr rdi, 0x3
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i64_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr x0, x0, #3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i64 %0, 3
    ret void
}

define void @lshr_i64_i64(i64 %0, i64 %1) {
; X64-LABEL: <lshr_i64_i64>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i64_i64>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr x0, x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i64 %0, %1
    ret void
}

define void @lshr_i37_3(i37 %0) {
; X64-LABEL: <lshr_i37_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movabs rax, 0x1fffffffff
; X64-NEXT:    and rdi, rax
; X64-NEXT:    shr rdi, 0x3
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i37_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ubfx x0, x0, #0, #37
; ARM64-NEXT:    lsr x0, x0, #3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i37 %0, 3
    ret void
}

define void @lshr_i21_3(i21 %0) {
; X64-LABEL: <lshr_i21_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    and edi, 0x1fffff
; X64-NEXT:    shr edi, 0x3
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i21_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ubfx w0, w0, #0, #21
; ARM64-NEXT:    lsr w0, w0, #3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i21 %0, 3
    ret void
}

define void @lshr_i21_i21(i21 %0, i21 %1) {
; X64-LABEL: <lshr_i21_i21>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    and edi, 0x1fffff
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr edi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i21_i21>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ubfx w0, w0, #0, #21
; ARM64-NEXT:    lsr w0, w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i21 %0, %1
    ret void
}

define void @lshr_i37_i37(i37 %0, i37 %1) {
; X64-LABEL: <lshr_i37_i37>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movabs rax, 0x1fffffffff
; X64-NEXT:    and rdi, rax
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i37_i37>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ubfx x0, x0, #0, #37
; ARM64-NEXT:    lsr x0, x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i37 %0, %1
    ret void
}

define i128 @lshr_i128_3(i128 %0) {
; X64-LABEL: <lshr_i128_3>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    shr rdi, 0x3
; X64-NEXT:    mov rax, rsi
; X64-NEXT:    shl rax, 0x3d
; X64-NEXT:    or rax, rdi
; X64-NEXT:    shr rsi, 0x3
; X64-NEXT:    mov rdx, rsi
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i128_3>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsl x2, x1, #61
; ARM64-NEXT:    lsr x0, x0, #3
; ARM64-NEXT:    lsr x1, x1, #3
; ARM64-NEXT:    orr x3, x0, x2
; ARM64-NEXT:    mov x0, x3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i128 %0, 3
    ret i128 %1
}

define i128 @lshr_i128_74(i128 %0) {
; X64-LABEL: <lshr_i128_74>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x40
; X64-NEXT:    shr rsi, 0xa
; X64-NEXT:    xor eax, eax
; X64-NEXT:    mov qword ptr [rbp - 0x38], rax
; X64-NEXT:    mov rax, rsi
; X64-NEXT:    mov rdx, qword ptr [rbp - 0x38]
; X64-NEXT:    add rsp, 0x40
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i128_74>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr x1, x1, #10
; ARM64-NEXT:    mov x2, xzr
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    mov x1, x2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i128 %0, 74
    ret i128 %1
}

define i128 @lshr_i128_128(i128 %0) {
; X64-LABEL: <lshr_i128_128>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    shr rdi, 0x0
; X64-NEXT:    mov rax, rsi
; X64-NEXT:    shl rax, 0x0
; X64-NEXT:    or rax, rdi
; X64-NEXT:    shr rsi, 0x0
; X64-NEXT:    mov rdx, rsi
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i128_128>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr x2, x1, #0
; ARM64-NEXT:    lsr x0, x0, #0
; ARM64-NEXT:    lsr x1, x1, #0
; ARM64-NEXT:    orr x3, x0, x2
; ARM64-NEXT:    mov x0, x3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i128 %0, 128
    ret i128 %1
}

define i128 @lshr_i128_i128(i128 %v, i128 %s) {
; X64-LABEL: <lshr_i128_i128>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    push rbx
; X64-NEXT:    nop dword ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x38
; X64-NEXT:    mov qword ptr [rbp - 0x38], rcx
; X64-NEXT:    mov ecx, edx
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    lea rax, [rsi + rsi]
; X64-NEXT:    mov ecx, edx
; X64-NEXT:    not cl
; X64-NEXT:    shl rax, cl
; X64-NEXT:    or rax, rdi
; X64-NEXT:    mov ecx, edx
; X64-NEXT:    shr rsi, cl
; X64-NEXT:    xor ebx, ebx
; X64-NEXT:    test dl, 0x40
; X64-NEXT:    cmovne rax, rsi
; X64-NEXT:    cmove rbx, rsi
; X64-NEXT:    mov rdx, rbx
; X64-NEXT:    add rsp, 0x38
; X64-NEXT:    pop rbx
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i128_i128>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsl x4, x1, #1
; ARM64-NEXT:    mvn w5, w2
; ARM64-NEXT:    lsr x0, x0, x2
; ARM64-NEXT:    tst x2, #0x40
; ARM64-NEXT:    lsl x4, x4, x5
; ARM64-NEXT:    lsr x5, x1, x2
; ARM64-NEXT:    orr x4, x4, x0
; ARM64-NEXT:    csel x1, xzr, x5, ne
; ARM64-NEXT:    csel x2, x5, x4, ne
; ARM64-NEXT:    mov x0, x2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %r = lshr i128 %v, %s
  ret i128 %r
}


define void @lshr_i64_no_salvage_imm(i64 %0) {
; X64-LABEL: <lshr_i64_no_salvage_imm>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    shr rax, 0x3
; X64-NEXT:    mov ecx, eax
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i64_no_salvage_imm>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr x1, x0, #3
; ARM64-NEXT:    lsr x0, x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i64 %0, 3
    %2 = lshr i64 %0, %1
    ret void
}

define void @lshr_i64_no_salvage_reg(i64 %0, i64 %1) {
; X64-LABEL: <lshr_i64_no_salvage_reg>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr rax, cl
; X64-NEXT:    mov ecx, eax
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i64_no_salvage_reg>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    lsr x1, x0, x1
; ARM64-NEXT:    lsr x0, x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i64 %0, %1
    %3 = lshr i64 %0, %2
    ret void
}

define void @lshr_i37_no_salvage_imm(i37 %0) {
; X64-LABEL: <lshr_i37_no_salvage_imm>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movabs rax, 0x1fffffffff
; X64-NEXT:    and rax, rdi
; X64-NEXT:    shr rax, 0x3
; X64-NEXT:    movabs rcx, 0x1fffffffff
; X64-NEXT:    and rdi, rcx
; X64-NEXT:    mov ecx, eax
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i37_no_salvage_imm>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ubfx x1, x0, #0, #37
; ARM64-NEXT:    lsr x1, x1, #3
; ARM64-NEXT:    ubfx x0, x0, #0, #37
; ARM64-NEXT:    lsr x0, x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %1 = lshr i37 %0, 3
    %2 = lshr i37 %0, %1
    ret void
}

define void @lshr_i37_no_salvage_reg(i37 %0, i37 %1) {
; X64-LABEL: <lshr_i37_no_salvage_reg>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movabs rax, 0x1fffffffff
; X64-NEXT:    and rax, rdi
; X64-NEXT:    mov ecx, esi
; X64-NEXT:    shr rax, cl
; X64-NEXT:    movabs rcx, 0x1fffffffff
; X64-NEXT:    and rdi, rcx
; X64-NEXT:    mov ecx, eax
; X64-NEXT:    shr rdi, cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <lshr_i37_no_salvage_reg>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ubfx x2, x0, #0, #37
; ARM64-NEXT:    lsr x2, x2, x1
; ARM64-NEXT:    ubfx x0, x0, #0, #37
; ARM64-NEXT:    lsr x0, x0, x2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
    %2 = lshr i37 %0, %1
    %3 = lshr i37 %0, %2
    ret void
}
