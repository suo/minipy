#include "Serialization.h"

namespace minipy {
std::string toString(OpCode c) {
  switch (c) {
#define DEFINE_STR(op)                  \
  case OpCode::op: \
    return #op;
    FORALL_OPCODES(DEFINE_STR)
#undef DEFINE_STR
  }
}

void dump(const CodeObject& co) {
  size_t offset = 0;
  for (const Instruction& instruction : co.instructions) {
    if (instruction.op == OpCode::LOAD_FAST ||
        instruction.op == OpCode::STORE_FAST) {
      fmt::print(
          "{:<8}{} ({})\n",
          offset++,
          instruction,
          co.varnames.at(instruction.arg1));
    } else if (
        instruction.op == OpCode::STORE_GLOBAL ||
        instruction.op == OpCode::LOAD_GLOBAL ||
        instruction.op == OpCode::LOAD_METHOD ||
        instruction.op == OpCode::LOAD_NAME ||
        instruction.op == OpCode::STORE_NAME ||
        instruction.op == OpCode::LOAD_ATTR) {
      fmt::print(
          "{:<8}{} ({})\n",
          offset++,
          instruction,
          co.names.at(instruction.arg1));
    } else if (instruction.op == OpCode::LOAD_CONST) {
      // TODO can't print obj yet
      //   fmt::print(
      //       "{:<8}{:20} {:<4} ({})\n",
      //       offset++,
      //       toString(instruction.op),
      //       instruction.arg1,
      //       constants.at(instruction.arg1));
    } else {
      fmt::print(
          "{:<8}{:20} {}\n",
          offset++,
          toString(instruction.op),
          instruction.arg1);
    }
  }
}

} // namespace minipy
