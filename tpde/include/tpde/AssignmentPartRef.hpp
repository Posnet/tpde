// SPDX-FileCopyrightText: 2024 Tobias Schwarz <tobias.schwarz@tum.de>
//
// SPDX-License-Identifier: LicenseRef-Proprietary
#pragma once

#include "tpde/RegisterFile.hpp"
#include "tpde/ValueAssignment.hpp"

#include <cstdint>

namespace tpde {

class AssignmentPartRef {
  ValueAssignment *va;
  uint32_t part;

  // note for how parts are structured:
  // |15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  // |  |   PS   |RV|  |IM|FA|  bank  |    reg_id    |
  //                         |      full_reg_id      |
  //
  // PS: 1 << PS = part size (TODO(ts): maybe swap with NP so that it can be
  //     extracted easier?)
  // RV: Register Valid
  // IM: Is the current register value not on the stack?
  // FA: Is the assignment a fixed assignment?
  //
  // RV + IM form a unit describing the following states:
  //  - !RV +  IM: value uninitialized (default state)
  //  -  RV +  IM: register dirty, must be spilled before evicting
  //  - !RV + !IM: register invalid, value stored only in stack slot
  //  -  RV + !IM: register identical to value in stack slot

public:
  AssignmentPartRef(ValueAssignment *va, const uint32_t part)
      : va(va), part(part) {}

  void reset() noexcept {
    va->parts[part] = 0;
    set_modified(true);
  }

  ValueAssignment *assignment() noexcept { return va; }

  [[nodiscard]] RegBank bank() const noexcept {
    return RegBank((va->parts[part] >> 5) & 0b111);
  }

  void set_bank(const RegBank bank) noexcept {
    assert(bank.id() <= 0b111);
    auto data = va->parts[part] & ~0b1110'0000;
    data |= bank.id() << 5;
    va->parts[part] = data;
  }

  [[nodiscard]] Reg get_reg() const noexcept {
    return Reg(va->parts[part] & 0xFF);
  }

  void set_reg(Reg reg) noexcept {
    assert(bank().id() == ((reg.id() >> 5) & 0b111));
    va->parts[part] = (va->parts[part] & 0xFF00) | reg.id();
  }

  [[nodiscard]] bool modified() const noexcept {
    return (va->parts[part] & (1u << 9)) != 0;
  }

  void set_modified(const bool val) noexcept {
    if (val) {
      va->parts[part] |= (1u << 9);
    } else {
      va->parts[part] &= ~(1u << 9);
    }
  }

  [[nodiscard]] bool fixed_assignment() const noexcept {
    return (va->parts[part] & (1u << 8)) != 0;
  }

  void set_fixed_assignment(const bool val) noexcept {
    if (val) {
      va->parts[part] |= (1u << 8);
    } else {
      va->parts[part] &= ~(1u << 8);
    }
  }

  [[nodiscard]] bool variable_ref() const noexcept { return va->variable_ref; }

  [[nodiscard]] bool is_stack_variable() const noexcept {
    return va->stack_variable;
  }

  [[nodiscard]] bool register_valid() const noexcept {
    return (va->parts[part] & (1u << 11)) != 0;
  }

  void set_register_valid(const bool val) noexcept {
    if (val) {
      va->parts[part] |= (1u << 11);
    } else {
      va->parts[part] &= ~(1u << 11);
    }
  }

  [[nodiscard]] bool stack_valid() const noexcept {
    return (va->parts[part] & (1u << 9)) == 0;
  }

  void set_stack_valid() noexcept { set_modified(false); }

  [[nodiscard]] uint32_t part_size() const noexcept {
    return 1u << ((va->parts[part] >> 12) & 0b111);
  }

  void set_part_size(const uint32_t part_size) noexcept {
    assert((part_size & (part_size - 1)) == 0);
    const uint32_t shift = util::cnt_tz(part_size);
    assert(shift <= 0b111);
    auto data = va->parts[part] & ~(0b111 << 12);
    data |= (shift << 12);
    va->parts[part] = data;
  }

  [[nodiscard]] int32_t frame_off() const noexcept {
    assert(!variable_ref());
    assert(va->frame_off != 0 && "attempt to access uninitialized stack slot");
    return va->frame_off + part_off();
  }

  [[nodiscard]] int32_t variable_stack_off() const noexcept {
    assert(variable_ref() && va->stack_variable);
    assert(part == 0);
    return va->frame_off;
  }

  [[nodiscard]] uint32_t variable_ref_data() const noexcept {
    assert(variable_ref() && !va->stack_variable);
    assert(part == 0);
    return va->var_ref_custom_idx;
  }

  [[nodiscard]] uint32_t part_off() const noexcept {
    return va->max_part_size * part;
  }
};

} // namespace tpde
