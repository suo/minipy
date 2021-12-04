#include <minipy/instruction.h>

#include <fmt/format.h>

namespace torch {
namespace jit {
namespace dynamic {
std::ostream& operator<<(std::ostream& os, const Instruction& instruction) {
  return os << fmt::format(
             "{:20} {:<4}", toString(instruction.op), instruction.arg1);
}
} // namespace dynamic
} // namespace jit
} // namespace torch
