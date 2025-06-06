; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

@basic_int = internal global i32 5, align 4
@global_int = global i32 5, align 4
@global_dso_local_int = dso_local global i32 5, align 4
@basic_array = dso_local global [20 x i32] zeroinitializer, align 16
@more_array = dso_local global [10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10], align 16
@func_ptr = dso_local global ptr @some_func, align 8
@global_ptr = dso_local global ptr getelementptr (i8, ptr @basic_array, i64 16), align 8

declare void @some_func(i32 noundef) #0

define i32 @load_basic_int() {
; X64-LABEL: <load_basic_int>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    lea rax, <load_basic_int+0x13>
; X64-NEXT:     R_X86_64_PC32 basic_int-0x4
; X64-NEXT:    mov ecx, dword ptr [rax]
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <load_basic_int>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_PREL_PG_HI21 basic_int
; ARM64-NEXT:    add x0, x0, #0x0
; ARM64-NEXT:     R_AARCH64_ADD_ABS_LO12_NC basic_int
; ARM64-NEXT:    ldr w1, [x0]
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %0 = load i32, ptr @basic_int
  ret i32 %0
}

define i32 @load_basic_int_twice() {
; X64-LABEL: <load_basic_int_twice>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    lea rax, <load_basic_int_twice+0x13>
; X64-NEXT:     R_X86_64_PC32 basic_int-0x4
; X64-NEXT:    mov ecx, dword ptr [rax]
; X64-NEXT:    mov edx, dword ptr [rax]
; X64-NEXT:    lea ecx, [rcx + rdx]
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <load_basic_int_twice>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_PREL_PG_HI21 basic_int
; ARM64-NEXT:    add x0, x0, #0x0
; ARM64-NEXT:     R_AARCH64_ADD_ABS_LO12_NC basic_int
; ARM64-NEXT:    ldr w1, [x0]
; ARM64-NEXT:    ldr w2, [x0]
; ARM64-NEXT:    add w2, w2, w1
; ARM64-NEXT:    mov w0, w2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %l0 = load i32, ptr @basic_int
  %l1 = load i32, ptr @basic_int
  %sum = add i32 %l0, %l1
  ret i32 %sum
}

define i32 @load_global_int() {
; X64-LABEL: <load_global_int>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <load_global_int+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL global_int-0x4
; X64-NEXT:    mov ecx, dword ptr [rax]
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <load_global_int>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE global_int
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC global_int
; ARM64-NEXT:    ldr w1, [x0]
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %l = load i32, ptr @global_int
  ret i32 %l
}

define i32 @load_global_dso_local_int() {
; X64-LABEL: <load_global_dso_local_int>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <load_global_dso_local_int+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL global_dso_local_int-0x4
; X64-NEXT:    mov ecx, dword ptr [rax]
; X64-NEXT:    mov eax, ecx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <load_global_dso_local_int>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE global_dso_local_int
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC global_dso_local_int
; ARM64-NEXT:    ldr w1, [x0]
; ARM64-NEXT:    mov w0, w1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  %l = load i32, ptr @global_dso_local_int
  ret i32 %l
}

define ptr @load_func_ptr() {
; X64-LABEL: <load_func_ptr>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <load_func_ptr+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL func_ptr-0x4
; X64-NEXT:    mov rcx, qword ptr [rax]
; X64-NEXT:    mov rax, rcx
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <load_func_ptr>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE func_ptr
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC func_ptr
; ARM64-NEXT:    ldr x1, [x0]
; ARM64-NEXT:    mov x0, x1
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  %0 = load ptr, ptr @func_ptr
  ret ptr %0
}

define void @store_global_ptr(ptr %0) {
; X64-LABEL: <store_global_ptr>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <store_global_ptr+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL global_ptr-0x4
; X64-NEXT:    mov qword ptr [rax], rdi
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <store_global_ptr>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x1, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE global_ptr
; ARM64-NEXT:    ldr x1, [x1]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC global_ptr
; ARM64-NEXT:    str x0, [x1]
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  store ptr %0, ptr @global_ptr
  ret void
}

define ptr @get_global() {
; X64-LABEL: <get_global>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    lea rax, <get_global+0x13>
; X64-NEXT:     R_X86_64_PC32 basic_int-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <get_global>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_PREL_PG_HI21 basic_int
; ARM64-NEXT:    add x0, x0, #0x0
; ARM64-NEXT:     R_AARCH64_ADD_ABS_LO12_NC basic_int
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  ret ptr @basic_int
}

define ptr @get_func1() {
; X64-LABEL: <get_func1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <get_func1+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL func_ptr-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <get_func1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE func_ptr
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC func_ptr
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  ret ptr @func_ptr
}

define ptr @get_func2() {
; X64-LABEL: <get_func2>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <get_func2+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL some_func-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <get_func2>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE some_func
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC some_func
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
entry:
  ret ptr @some_func
}

define {ptr, ptr} @get_struct1() {
; X64-LABEL: <get_struct1>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <get_struct1+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL func_ptr-0x4
; X64-NEXT:    lea rdx, <get_struct1+0x1a>
; X64-NEXT:     R_X86_64_PC32 basic_int-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <get_struct1>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE func_ptr
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC func_ptr
; ARM64-NEXT:    adrp x1, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_PREL_PG_HI21 basic_int
; ARM64-NEXT:    add x1, x1, #0x0
; ARM64-NEXT:     R_AARCH64_ADD_ABS_LO12_NC basic_int
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  ret {ptr, ptr} { ptr @func_ptr, ptr @basic_int }
}

define {ptr, ptr} @get_struct2() {
; X64-LABEL: <get_struct2>:
; X64:         push rbp
; X64-NEXT:    mov rbp, rsp
; X64-NEXT:    nop word ptr [rax + rax]
; X64-NEXT:    sub rsp, 0x30
; X64-NEXT:    mov rax, qword ptr <get_struct2+0x13>
; X64-NEXT:     R_X86_64_GOTPCREL some_func-0x4
; X64-NEXT:    mov rdx, qword ptr <get_struct2+0x1a>
; X64-NEXT:     R_X86_64_GOTPCREL get_struct2-0x4
; X64-NEXT:    add rsp, 0x30
; X64-NEXT:    pop rbp
; X64-NEXT:    ret
;
; ARM64-LABEL: <get_struct2>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    adrp x0, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE some_func
; ARM64-NEXT:    ldr x0, [x0]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC some_func
; ARM64-NEXT:    adrp x1, 0x0 <.text>
; ARM64-NEXT:     R_AARCH64_ADR_GOT_PAGE get_struct2
; ARM64-NEXT:    ldr x1, [x1]
; ARM64-NEXT:     R_AARCH64_LD64_GOT_LO12_NC get_struct2
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  ret {ptr, ptr} { ptr @some_func, ptr @get_struct2 }
}

