#pragma once

#include <fmt/format.h>
#include "minipy/common/instruction.h"
#include "minipy/interpreter/types.h"

namespace minipy {
std::string toString(OpCode c);
void dump(const CodeObject& co);
} // namespace minipy

template <>
struct fmt::formatter<minipy::Instruction> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.end();
  }

  template <typename FormatContext>
  auto format(const minipy::Instruction& inst, FormatContext& ctx)
      -> decltype(ctx.out()) {
    return format_to(
        ctx.out(), "{:20} {:<4}", minipy::toString(inst.op), inst.arg1);
  }
};
