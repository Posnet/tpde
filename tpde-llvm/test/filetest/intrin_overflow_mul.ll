; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

declare {i8, i1} @llvm.smul.with.overflow.i8(i8, i8)
declare {i16, i1} @llvm.smul.with.overflow.i16(i16, i16)
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32)
declare {i64, i1} @llvm.smul.with.overflow.i64(i64, i64)
declare {i128, i1} @llvm.smul.with.overflow.i128(i128, i128)

declare {i8, i1} @llvm.umul.with.overflow.i8(i8, i8)
declare {i16, i1} @llvm.umul.with.overflow.i16(i16, i16)
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32)
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64)
declare {i128, i1} @llvm.umul.with.overflow.i128(i128, i128)

define i8 @umul_i8_0(i8 %0, i8 %1) {
; X64-LABEL: <umul_i8_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    mul sil
; X64-NEXT:    seto cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i8_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    and w1, w1, #0xff
; ARM64-NEXT:    and w0, w0, #0xff
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    tst w1, #0xff00
; ARM64-NEXT:    and x2, x1, #0xff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i8, i1} @llvm.umul.with.overflow.i8(i8 %0, i8 %1)
  %3 = extractvalue {i8, i1} %2, 0
  ret i8 %3
}

define i1 @umul_i8_1(i8 %0, i8 %1) {
; X64-LABEL: <umul_i8_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    mul sil
; X64-NEXT:    seto cl
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i8_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    and w1, w1, #0xff
; ARM64-NEXT:    and w0, w0, #0xff
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    tst w1, #0xff00
; ARM64-NEXT:    and x2, x1, #0xff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i8, i1} @llvm.umul.with.overflow.i8(i8 %0, i8 %1)
  %3 = extractvalue {i8, i1} %2, 1
  ret i1 %3
}

define i16 @umul_i16_0(i16 %0, i16 %1) {
; X64-LABEL: <umul_i16_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    mul si
; X64-NEXT:    seto cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i16_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    and w1, w1, #0xffff
; ARM64-NEXT:    and w0, w0, #0xffff
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    tst w1, #0xffff0000
; ARM64-NEXT:    and x2, x1, #0xffff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i16, i1} @llvm.umul.with.overflow.i16(i16 %0, i16 %1)
  %3 = extractvalue {i16, i1} %2, 0
  ret i16 %3
}

define i1 @umul_i16_1(i16 %0, i16 %1) {
; X64-LABEL: <umul_i16_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    mul si
; X64-NEXT:    seto cl
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i16_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    and w1, w1, #0xffff
; ARM64-NEXT:    and w0, w0, #0xffff
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    tst w1, #0xffff0000
; ARM64-NEXT:    and x2, x1, #0xffff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i16, i1} @llvm.umul.with.overflow.i16(i16 %0, i16 %1)
  %3 = extractvalue {i16, i1} %2, 1
  ret i1 %3
}

define i32 @umul_i32_0(i32 %0, i32 %1) {
; X64-LABEL: <umul_i32_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    mul esi
; X64-NEXT:    seto cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i32_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    umull x0, w0, w1
; ARM64-NEXT:    tst x0, #0xffffffff00000000
; ARM64-NEXT:    mov w1, w0
; ARM64-NEXT:    cset w2, ne
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue {i32, i1} %2, 0
  ret i32 %3
}

define i1 @umul_i32_1(i32 %0, i32 %1) {
; X64-LABEL: <umul_i32_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    mul esi
; X64-NEXT:    seto cl
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i32_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    umull x0, w0, w1
; ARM64-NEXT:    tst x0, #0xffffffff00000000
; ARM64-NEXT:    mov w1, w0
; ARM64-NEXT:    cset w2, ne
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue {i32, i1} %2, 1
  ret i1 %3
}

define i64 @umul_i64_0(i64 %0, i64 %1) {
; X64-LABEL: <umul_i64_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    mul rsi
; X64-NEXT:    seto cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i64_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    umulh x2, x0, x1
; ARM64-NEXT:    mul x0, x0, x1
; ARM64-NEXT:    cmp xzr, x2
; ARM64-NEXT:    cset w1, ne
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %0, i64 %1)
  %3 = extractvalue {i64, i1} %2, 0
  ret i64 %3
}

define i1 @umul_i64_1(i64 %0, i64 %1) {
; X64-LABEL: <umul_i64_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    mul rsi
; X64-NEXT:    seto cl
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <umul_i64_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    umulh x2, x0, x1
; ARM64-NEXT:    mul x0, x0, x1
; ARM64-NEXT:    cmp xzr, x2
; ARM64-NEXT:    cset w1, ne
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %0, i64 %1)
  %3 = extractvalue {i64, i1} %2, 1
  ret i1 %3
}



define i8 @smul_i8_0(i8 %0, i8 %1) {
; X64-LABEL: <smul_i8_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    imul sil
; X64-NEXT:    seto cl
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i8_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w1, w1
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    cmp w1, w1, sxtb
; ARM64-NEXT:    and x2, x1, #0xff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i8, i1} @llvm.smul.with.overflow.i8(i8 %0, i8 %1)
  %3 = extractvalue {i8, i1} %2, 0
  ret i8 %3
}

define i1 @smul_i8_1(i8 %0, i8 %1) {
; X64-LABEL: <smul_i8_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor ecx, ecx
; X64-NEXT:    mov eax, edi
; X64-NEXT:    imul sil
; X64-NEXT:    seto cl
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i8_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxtb w1, w1
; ARM64-NEXT:    sxtb w0, w0
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    cmp w1, w1, sxtb
; ARM64-NEXT:    and x2, x1, #0xff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i8, i1} @llvm.smul.with.overflow.i8(i8 %0, i8 %1)
  %3 = extractvalue {i8, i1} %2, 1
  ret i1 %3
}

define i16 @smul_i16_0(i16 %0, i16 %1) {
; X64-LABEL: <smul_i16_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor eax, eax
; X64-NEXT:    imul di, si
; X64-NEXT:    seto al
; X64-NEXT:    mov eax, edi
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i16_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w1, w1
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    cmp w1, w1, sxth
; ARM64-NEXT:    and x2, x1, #0xffff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i16, i1} @llvm.smul.with.overflow.i16(i16 %0, i16 %1)
  %3 = extractvalue {i16, i1} %2, 0
  ret i16 %3
}

define i1 @smul_i16_1(i16 %0, i16 %1) {
; X64-LABEL: <smul_i16_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor eax, eax
; X64-NEXT:    imul di, si
; X64-NEXT:    seto al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i16_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    sxth w1, w1
; ARM64-NEXT:    sxth w0, w0
; ARM64-NEXT:    mul w1, w0, w1
; ARM64-NEXT:    cmp w1, w1, sxth
; ARM64-NEXT:    and x2, x1, #0xffff
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i16, i1} @llvm.smul.with.overflow.i16(i16 %0, i16 %1)
  %3 = extractvalue {i16, i1} %2, 1
  ret i1 %3
}

define i32 @smul_i32_0(i32 %0, i32 %1) {
; X64-LABEL: <smul_i32_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor eax, eax
; X64-NEXT:    imul edi, esi
; X64-NEXT:    seto al
; X64-NEXT:    mov eax, edi
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i32_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    smull x0, w0, w1
; ARM64-NEXT:    cmp x0, w0, sxtw
; ARM64-NEXT:    mov w1, w0
; ARM64-NEXT:    cset w2, ne
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue {i32, i1} %2, 0
  ret i32 %3
}

define i1 @smul_i32_1(i32 %0, i32 %1) {
; X64-LABEL: <smul_i32_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor eax, eax
; X64-NEXT:    imul edi, esi
; X64-NEXT:    seto al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i32_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    smull x0, w0, w1
; ARM64-NEXT:    cmp x0, w0, sxtw
; ARM64-NEXT:    mov w1, w0
; ARM64-NEXT:    cset w2, ne
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue {i32, i1} %2, 1
  ret i1 %3
}

define i64 @smul_i64_0(i64 %0, i64 %1) {
; X64-LABEL: <smul_i64_0>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor eax, eax
; X64-NEXT:    imul rdi, rsi
; X64-NEXT:    seto al
; X64-NEXT:    mov rax, rdi
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i64_0>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mul x2, x0, x1
; ARM64-NEXT:    smulh x0, x0, x1
; ARM64-NEXT:    mov x1, x2
; ARM64-NEXT:    cmp x0, x2, asr #63
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %0, i64 %1)
  %3 = extractvalue {i64, i1} %2, 0
  ret i64 %3
}

define i1 @smul_i64_1(i64 %0, i64 %1) {
; X64-LABEL: <smul_i64_1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    xor eax, eax
; X64-NEXT:    imul rdi, rsi
; X64-NEXT:    seto al
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <smul_i64_1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    mul x2, x0, x1
; ARM64-NEXT:    smulh x0, x0, x1
; ARM64-NEXT:    mov x1, x2
; ARM64-NEXT:    cmp x0, x2, asr #63
; ARM64-NEXT:    cset w3, ne
; ARM64-NEXT:    mov w0, w3
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %2 = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %0, i64 %1)
  %3 = extractvalue {i64, i1} %2, 1
  ret i1 %3
}
