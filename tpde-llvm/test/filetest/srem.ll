; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

define i8 @srem_i8_1(i8 %0) {
; X64-LABEL: <srem_i8_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x1
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    mov x1, #0x1 // =1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i8 %0, 1
  ret i8 %1
}

define i8 @srem_i8_-1(i8 %0) {
; X64-LABEL: <srem_i8_-1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0xffffffff
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_-1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    mov x1, #-0x1 // =-1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i8 %0, -1
  ret i8 %1
}

define i8 @srem_i8_28(i8 %0) {
; X64-LABEL: <srem_i8_28>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x1c
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_28>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    mov x1, #0x1c // =28
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i8 %0, 28
  ret i8 %1
}

define i8 @srem_i8_i8(i8 %0, i8 %1) {
; X64-LABEL: <srem_i8_i8>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    movsx esi, sil
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_i8>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    sxtb w1, w1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i8 %0, %1
  ret i8 %2
}

define i8 @srem_i8_32(i8 %0) {
; X64-LABEL: <srem_i8_32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x20
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    mov x1, #0x20 // =32
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i8 %0, 32
  ret i8 %1
}

define i16 @srem_i16_1(i16 %0) {
; X64-LABEL: <srem_i16_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, di
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x1
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    mov x1, #0x1 // =1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i16 %0, 1
  ret i16 %1
}

define i16 @srem_i16_-1(i16 %0) {
; X64-LABEL: <srem_i16_-1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, di
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0xffffffff
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_-1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    mov x1, #-0x1 // =-1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i16 %0, -1
  ret i16 %1
}

define i16 @srem_i16_28(i16 %0) {
; X64-LABEL: <srem_i16_28>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, di
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x1c
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_28>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    mov x1, #0x1c // =28
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i16 %0, 28
  ret i16 %1
}

define i16 @srem_i16_32(i16 %0) {
; X64-LABEL: <srem_i16_32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, di
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x20
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    mov x1, #0x20 // =32
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i16 %0, 32
  ret i16 %1
}

define i16 @srem_i16_i16(i16 %0, i16 %1) {
; X64-LABEL: <srem_i16_i16>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, di
; X64-NEXT:    movsx esi, si
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_i16>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    sxth w1, w1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i16 %0, %1
  ret i16 %2
}

define i32 @srem_i32_1(i32 %0) {
; X64-LABEL: <srem_i32_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x1
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0x1 // =1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i32 %0, 1
  ret i32 %1
}

define i32 @srem_i32_-1(i32 %0) {
; X64-LABEL: <srem_i32_-1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0xffffffff
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_-1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0xffffffff // =4294967295
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i32 %0, -1
  ret i32 %1
}

define i32 @srem_i32_28(i32 %0) {
; X64-LABEL: <srem_i32_28>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x1c
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_28>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0x1c // =28
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i32 %0, 28
  ret i32 %1
}

define i32 @srem_i32_32(i32 %0) {
; X64-LABEL: <srem_i32_32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    mov ecx, 0x20
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0x20 // =32
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i32 %0, 32
  ret i32 %1
}

define i32 @srem_i32_i32(i32 %0, i32 %1) {
; X64-LABEL: <srem_i32_i32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_i32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i32 %0, %1
  ret i32 %2
}

define i64 @srem_i64_1(i64 %0) {
; X64-LABEL: <srem_i64_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    mov rcx, 0x1
; X64-NEXT:    idiv rcx
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0x1 // =1
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i64 %0, 1
  ret i64 %1
}

define i64 @srem_i64_-1(i64 %0) {
; X64-LABEL: <srem_i64_-1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    mov rcx, -0x1
; X64-NEXT:    idiv rcx
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_-1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #-0x1 // =-1
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i64 %0, -1
  ret i64 %1
}

define i64 @srem_i64_28(i64 %0) {
; X64-LABEL: <srem_i64_28>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    mov rcx, 0x1c
; X64-NEXT:    idiv rcx
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_28>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0x1c // =28
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i64 %0, 28
  ret i64 %1
}

define i64 @srem_i64_32(i64 %0) {
; X64-LABEL: <srem_i64_32>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    mov rcx, 0x20
; X64-NEXT:    idiv rcx
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_32>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mov x1, #0x20 // =32
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %1 = srem i64 %0, 32
  ret i64 %1
}

define i64 @srem_i64_i64(i64 %0, i64 %1) {
; X64-LABEL: <srem_i64_i64>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    idiv rsi
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_i64>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i64 %0, %1
  ret i64 %2
}

define i8 @srem_i8_salvage(i8 %0, i8 %1) {
; X64-LABEL: <srem_i8_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    movsx esi, sil
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    sxtb w1, w1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i8 %0, %1
  ret i8 %2
}

define i16 @srem_i16_salvage(i16 %0, i16 %1) {
; X64-LABEL: <srem_i16_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx edi, di
; X64-NEXT:    movsx esi, si
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    sxth w1, w1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i16 %0, %1
  ret i16 %2
}

define i32 @srem_i32_salvage(i32 %0, i32 %1) {
; X64-LABEL: <srem_i32_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i32 %0, %1
  ret i32 %2
}

define i64 @srem_i64_salvage(i64 %0, i64 %1) {
; X64-LABEL: <srem_i64_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    idiv rsi
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i64 %0, %1
  ret i64 %2
}

define i8 @srem_i8_no_salvage(i8 %0, i8 %1) {
; X64-LABEL: <srem_i8_no_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx eax, dil
; X64-NEXT:    movsx esi, sil
; X64-NEXT:    mov rcx, rax
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    movsx edi, dil
; X64-NEXT:    movsx edx, dl
; X64-NEXT:    mov rcx, rdx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i8_no_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w2, w0
; ARM64-NEXT:    sxtb w1, w1
; ARM64-NEXT:    sdiv w3, w2, w1
; ARM64-NEXT:    msub w1, w3, w1, w2
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    sxtb w1, w1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i8 %0, %1
  %3 = srem i8 %0, %2
  ret i8 %3
}

define i16 @srem_i16_no_salvage(i16 %0, i16 %1) {
; X64-LABEL: <srem_i16_no_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    movsx eax, di
; X64-NEXT:    movsx esi, si
; X64-NEXT:    mov rcx, rax
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    movsx edi, di
; X64-NEXT:    movsx edx, dx
; X64-NEXT:    mov rcx, rdx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv ecx
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i16_no_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w2, w0
; ARM64-NEXT:    sxth w1, w1
; ARM64-NEXT:    sdiv w3, w2, w1
; ARM64-NEXT:    msub w1, w3, w1, w2
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    sxth w1, w1
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i16 %0, %1
  %3 = srem i16 %0, %2
  ret i16 %3
}

define i32 @srem_i32_no_salvage(i32 %0, i32 %1) {
; X64-LABEL: <srem_i32_no_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv esi
; X64-NEXT:    mov dword ptr [rbp - 0x2c], edx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    cdq
; X64-NEXT:    idiv dword ptr [rbp - 0x2c]
; X64-NEXT:    mov eax, edx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i32_no_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    sdiv w2, w0, w1
; ARM64-NEXT:    msub w1, w2, w1, w0
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i32 %0, %1
  %3 = srem i32 %0, %2
  ret i32 %3
}

define i64 @srem_i64_no_salvage(i64 %0, i64 %1) {
; X64-LABEL: <srem_i64_no_salvage>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    idiv rsi
; X64-NEXT:    mov qword ptr [rbp - 0x30], rdx
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    cqo
; X64-NEXT:    idiv qword ptr [rbp - 0x30]
; X64-NEXT:    mov rax, rdx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i64_no_salvage>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    sdiv x2, x0, x1
; ARM64-NEXT:    msub x1, x2, x1, x0
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = srem i64 %0, %1
  %3 = srem i64 %0, %2
  ret i64 %3
}

define i128 @srem_i128(i128 %0, i128 %1) {
; X64-LABEL: <srem_i128>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:  <L0>:
; X64-NEXT:    call <L0>
; X64-NEXT:     R_X86_64_PLT32 __modti3-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <srem_i128>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    bl 0x910 <srem_i128+0x10>
; ARM64-NEXT:     R_AARCH64_CALL26 __modti3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %r = srem i128 %0, %1
  ret i128 %r
}
