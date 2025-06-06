; NOTE: Assertions have been autogenerated by test/update_tpde_llc_test_checks.py UTC_ARGS: --version 5
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=aarch64 %s | %objdump | FileCheck %s -check-prefixes=ARM64

define void @inlineasm() {
; ARM64-LABEL: <inlineasm>:
; ARM64:         sub sp, sp, #0xa0
; ARM64-NEXT:    stp x29, x30, [sp]
; ARM64-NEXT:    mov x29, sp
; ARM64-NEXT:    nop
; ARM64-NEXT:    ldp x29, x30, [sp]
; ARM64-NEXT:    add sp, sp, #0xa0
; ARM64-NEXT:    ret
  call void asm "", ""()
  call void asm sideeffect "", ""()
  call void asm unwind "", ""()
  call void asm sideeffect "", "~{memory}"()
  call void asm unwind "", "~{memory}"()
  ret void
}
