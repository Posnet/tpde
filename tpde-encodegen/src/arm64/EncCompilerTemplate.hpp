// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

// TODO(ts): I'd really like to keep these as normal headers in the source tree
// and generate this header
// TODO(ts): then also work with placeholders like
// -- DECLS_HERE --
// or
// -- IMPL_HERE --

namespace tpde_encgen::arm64 {
static constexpr inline char ENCODER_TEMPLATE_BEGIN[] =
    R"(// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception AND CC0-1.0

// NOTE: This file is autogenerated by tpde-encodegen. Please DO NOT edit this file
// as all changes will be overwritten once the file is generated again.
// NOTE: Some parts of this file are subject to the default license of TPDE
// and only the autogenerated code falls under the CC0-1.0 license

// SPDX-SnippetBegin
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <variant>

#include "tpde/base.hpp"
#include "tpde/arm64/CompilerA64.hpp"

// Helper macros for assembling in the compiler
#if defined(ASMD)
    #error Got definition for ASM macros from somewhere else. Maybe you included compilers for multiple architectures?
#endif

#define ASMD(...) ASMC(this->derived(), __VA_ARGS__)

namespace tpde_encodegen {

using namespace tpde;

template <typename Adaptor,
          typename Derived,
          template <typename, typename, typename>
          typename BaseTy,
          typename Config>
struct EncodeCompiler {
    using CompilerA64  = tpde::a64::CompilerA64<Adaptor, Derived, BaseTy, Config>;
    using ScratchReg   = typename CompilerA64::ScratchReg;
    using AsmReg       = typename CompilerA64::AsmReg;
    using ValuePartRef = typename CompilerA64::ValuePartRef;
    using GenericValuePart = typename CompilerA64::GenericValuePart;
    using Assembler    = typename CompilerA64::Assembler;
    using Label        = typename Assembler::Label;
    using SymRef       = typename Assembler::SymRef;

    std::optional<u64> encodeable_as_shiftimm(GenericValuePart &gv, unsigned size) const noexcept {
        if (gv.is_imm() && gv.imm_size() <= 8) {
            return gv.imm64() & (size - 1);
        }
        return std::nullopt;
    }
    std::optional<u64> encodeable_as_immarith(GenericValuePart &gv) const noexcept {
        if (gv.is_imm() && gv.imm_size() <= 8) {
            u64 val = gv.imm64();
            val     = static_cast<i64>(val) < 0 ? -val : val;
            if ((val & 0xfff) == val || (val & 0xff'f000) == val) {
                return gv.imm64();
            }
        }
        return std::nullopt;
    }
    std::optional<u64> encodeable_as_immlogical(GenericValuePart &gv, bool inv) const noexcept {
        if (gv.is_imm()) {
            u64 imm = gv.imm64() ^ (inv ? ~u64{0} : 0);
            if (gv.imm_size() == 8 && de64_ANDxi(DA_GP(0), DA_GP(0), imm))
                return imm;
            if (gv.imm_size() <= 4 && de64_ANDwi(DA_GP(0), DA_GP(0), imm))
                return imm;
        }
        return std::nullopt;
    }
    std::optional<std::pair<AsmReg, u64>> encodeable_with_mem_uoff12(GenericValuePart &gv, u64 off, unsigned shift) noexcept;

    void   try_salvage_or_materialize(GenericValuePart &gv,
                                             ScratchReg     &dst_scratch,
                                             u8              bank,
                                             u32             size) noexcept;

    CompilerA64 *derived() noexcept {
        return static_cast<CompilerA64 *>(static_cast<Derived *>(this));
    }

    const CompilerA64 *derived() const noexcept {
        return static_cast<const CompilerA64 *>(
            static_cast<const Derived *>(this));
    }

    struct FixedRegBackup {
        ScratchReg  scratch;
        ValLocalIdx local_idx;
        u32         part;
        u32         lock_count;
    };

    void scratch_alloc_specific(AsmReg                              reg,
                                ScratchReg                         &scratch,
                                std::initializer_list<GenericValuePart *> operands,
                                FixedRegBackup &backup_reg) noexcept;

    void scratch_check_fixed_backup(ScratchReg     &scratch,
                                    FixedRegBackup &backup_reg,
                                    bool            is_ret_reg) noexcept;

    void reset() noexcept {
        symbols.fill({});
    }

// SPDX-SnippetEnd
// SPDX-SnippetBegin
// SPDX-License-Identifier: CC0-1.0
// clang-format off
)";

static constexpr inline char ENCODER_TEMPLATE_END[] = R"(
};
// SPDX-SnippetEnd
)";

static constexpr inline char ENCODER_IMPL_TEMPLATE_BEGIN[] = R"(

// SPDX-SnippetBegin
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format on
template <typename Adaptor,
          typename Derived,
          template <typename, typename, typename>
          class BaseTy,
          typename Config>
std::optional<std::pair<typename EncodeCompiler<Adaptor, Derived, BaseTy, Config>::AsmReg, u64>> EncodeCompiler<Adaptor, Derived, BaseTy, Config>::
    encodeable_with_mem_uoff12(GenericValuePart &gv, u64 off, unsigned shift) noexcept {
    auto *expr = std::get_if<typename GenericValuePart::Expr>(&gv.state);
    if (!expr || !expr->has_base()) {
        return std::nullopt;
    }

    u64 res_off = expr->disp + off;
    if (res_off >= (u64{0x1000} << shift) || (res_off & ((1 << shift) - 1))) {
        return std::nullopt;
    }
    if (res_off == 0 && expr->has_index()) {
        // In this case, try index encoding.
        return std::nullopt;
    }

    if (!expr->has_index()) {
        return std::make_pair(expr->base_reg(), res_off);
    }
    if ((expr->scale & (expr->scale - 1)) != 0) {
        return std::nullopt;
    }

    ScratchReg scratch{derived()};
    AsmReg base_reg = expr->base_reg();
    AsmReg index_reg = expr->index_reg();
    if (std::holds_alternative<ScratchReg>(expr->base)) {
        scratch = std::move(std::get<ScratchReg>(expr->base));
    } else if (std::holds_alternative<ScratchReg>(expr->index)) {
        scratch = std::move(std::get<ScratchReg>(expr->index));
    } else {
        (void)scratch.alloc_gp();
    }
    const auto scale_shift = __builtin_ctzl(expr->scale);
    AsmReg dst = scratch.cur_reg();
    ASMD(ADDx_lsl, dst, base_reg, index_reg, scale_shift);
    if (expr->disp != 0) {
        expr->base = std::move(scratch);
        expr->index = AsmReg::make_invalid();
        expr->scale = 0;
    } else {
        gv.state = std::move(scratch);
    }

    return std::make_pair(dst, res_off);
}

template <typename Adaptor,
          typename Derived,
          template <typename, typename, typename>
          class BaseTy,
          typename Config>
void EncodeCompiler<Adaptor, Derived, BaseTy, Config>::
    try_salvage_or_materialize(GenericValuePart &gv,
                               ScratchReg     &dst_scratch,
                               u8              bank,
                               u32             size) noexcept {
    AsmReg reg = derived()->gval_as_reg_reuse(gv, dst_scratch);
    if (!dst_scratch.has_reg()) {
        dst_scratch.alloc(RegBank(bank));
    }
    if (dst_scratch.cur_reg() != reg) {
        derived()->mov(dst_scratch.cur_reg(), reg, size);
    }
}

template <typename Adaptor,
          typename Derived,
          template <typename, typename, typename>
          class BaseTy,
          typename Config>
void EncodeCompiler<Adaptor, Derived, BaseTy, Config>::scratch_alloc_specific(
    AsmReg                              reg,
    ScratchReg                         &scratch,
    std::initializer_list<GenericValuePart *> operands,
    FixedRegBackup                     &backup_reg) noexcept {
    if (!derived()->register_file.is_fixed(reg)) [[likely]] {
        scratch.alloc_specific(reg);
        return;
    }

    const auto bank = derived()->register_file.reg_bank(reg);
    if (bank != Config::GP_BANK) {
        // TODO(ts): need to know the size
        TPDE_FATAL("fixed non-gp regs not supported");
    }

    const auto alloc_backup = [this, &backup_reg, &scratch, reg, bank]() {
        const auto bak_reg    = backup_reg.scratch.alloc(bank);
        auto      &reg_file   = derived()->register_file;
        auto      &assignment = reg_file.assignments[reg.id()];
        backup_reg.local_idx  = assignment.local_idx;
        backup_reg.part       = assignment.part;
        backup_reg.lock_count = reg_file.lock_counts[reg.id()];

        assignment.local_idx  = CompilerA64::INVALID_VAL_LOCAL_IDX;
        assignment.part       = 0;
        reg_file.lock_counts[reg.id()] = 0;

        assert(!scratch.has_reg());
        scratch.force_set_reg(reg);

        ASMD(MOV64rr, bak_reg, reg);
    };

    // check if one of the operands holds the fixed register
    for (auto *op_ptr : operands) {
        auto &op = op_ptr->state;
        if (std::holds_alternative<ScratchReg>(op)) {
            auto &op_scratch = std::get<ScratchReg>(op);
            if (op_scratch.cur_reg() == reg) {
                scratch = std::move(op_scratch);
                op_scratch.alloc(bank);
                ASMD(MOVx, op_scratch.cur_reg(), reg);
                return;
            }
            continue;
        }

        if (std::holds_alternative<ValuePartRef>(op)) {
            auto &op_ref = std::get<ValuePartRef>(op);
            if (op_ref.has_assignment()) {
                assert(!op_ref.has_reg());
                const auto ap = op_ref.assignment();
                if (ap.register_valid()) {
                    assert(ap.get_reg() != reg);
                }
            }
            continue;
        }

        if (std::holds_alternative<typename GenericValuePart::Expr>(op)) {
            auto &expr = std::get<typename GenericValuePart::Expr>(op);
            if (expr.base_reg() == reg) {
                if (std::holds_alternative<ScratchReg>(expr.base)) {
                    auto &op_scratch = std::get<ScratchReg>(expr.base);
                    scratch          = std::move(op_scratch);
                    op_scratch.alloc(bank);
                    ASMD(MOVx, op_scratch.cur_reg(), reg);
                } else {
                    alloc_backup();
                    expr.base = backup_reg.scratch.cur_reg();
                }
                return;
            }
            if (expr.scale != 0 && expr.index_reg() == reg) {
                if (std::holds_alternative<ScratchReg>(expr.index)) {
                    auto &op_scratch = std::get<ScratchReg>(expr.index);
                    scratch          = std::move(op_scratch);
                    op_scratch.alloc(bank);
                    ASMD(MOVx, op_scratch.cur_reg(), reg);
                } else {
                    alloc_backup();
                    expr.index = backup_reg.scratch.cur_reg();
                }
                return;
            }
            continue;
        }
    }

    // otherwise temporarily store it somewhere else
    alloc_backup();
    return;
}

template <typename Adaptor,
          typename Derived,
          template <typename, typename, typename>
          class BaseTy,
          typename Config>
void EncodeCompiler<Adaptor, Derived, BaseTy, Config>::
    scratch_check_fixed_backup(ScratchReg     &scratch,
                               FixedRegBackup &backup_reg,
                               const bool      is_ret_reg) noexcept {
    if (!backup_reg.scratch.has_reg()) [[likely]] {
        return;
    }

    assert(scratch.has_reg());
    auto &reg_file        = derived()->register_file;
    auto &assignment      = reg_file.assignments[scratch.cur_reg().id()];
    assignment.local_idx  = backup_reg.local_idx;
    assignment.part       = backup_reg.part;
    reg_file.lock_counts[scratch.cur_reg().id()] = backup_reg.lock_count;

    assert(reg_file.reg_bank(scratch.cur_reg()) == 0);
    if (is_ret_reg) {
        // TODO(ts): allocate another scratch? Though at this point the scratch
        // regs have not been released yet so we might need to spill...

        // need to switch around backup and reg so it can be returned as a
        // ScratchReg
        assert(false);
        // ASMD(XCHG64rr, scratch.cur_reg, backup_reg.scratch.cur_reg);
        // scratch.cur_reg            = backup_reg.scratch.cur_reg;
        // backup_reg.scratch.cur_reg = AsmReg::make_invalid();
    } else {
        ASMD(MOVx, scratch.cur_reg(), backup_reg.scratch.cur_reg());

        scratch.force_set_reg(AsmReg::make_invalid());
        backup_reg.scratch.reset();
    }
}

// clang-format off
// SPDX-SnippetEnd
// SPDX-SnippetBegin
// SPDX-License-Identifier: CC0-1.0
)";


static constexpr inline char ENCODER_IMPL_TEMPLATE_END[] = R"(
} // namespace tpde_encodegen

#undef ASMD
// SPDX-SnippetEnd
)";
} // namespace tpde_encgen::arm64
