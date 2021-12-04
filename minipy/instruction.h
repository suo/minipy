#pragma once

#include <string>

namespace torch {
namespace jit {
namespace dynamic {
// Opcode notes:
// - Based on opcoes as of py 3.8.3
// - relative jump opcodes omitted, seems like the primary purpose is reducing
// operand size:
// https://mail.python.org/pipermail/python-dev/2008-March/078104.html

#define FORALL_OPCODES(_) \
  _(LOAD_FAST)            \
  _(LOAD_NAME)            \
  _(STORE_NAME)           \
  _(LOAD_GLOBAL)          \
  _(LOAD_METHOD)          \
  _(CALL_METHOD)          \
  _(LOAD_CONST)           \
  _(STORE_FAST)           \
  _(STORE_GLOBAL)         \
  _(LOAD_ATTR)            \
  _(STORE_ATTR)           \
  _(CALL_FUNCTION)        \
  _(MAKE_FUNCTION)        \
  _(RETURN_VALUE)         \
  _(LOAD_BUILD_CLASS)     \
  _(POP_JUMP_IF_FALSE)    \
  _(POP_JUMP_IF_TRUE)     \
  _(JUMP_ABSOLUTE)        \
  _(COMPARE_OP)           \
  _(JUMP_IF_FALSE_OR_POP) \
  _(JUMP_IF_TRUE_OR_POP)  \
  _(BINARY_ADD)           \
  _(BINARY_MODULO)        \
  _(BINARY_FLOOR_DIVIDE)  \
  _(BINARY_TRUE_DIVIDE)   \
  _(GET_ITER)             \
  _(FOR_ITER)             \
  _(FOR_LOOP)             \
  _(UNARY_NOT)            \
  _(INPLACE_ADD)          \
  _(RAISE_VARARG)         \
  _(BUILD_LIST)           \
  _(POP_TOP)

enum class OpCode {
#define DEFINE_OP(op) op,
  FORALL_OPCODES(DEFINE_OP)
#undef DEFINE_OP
};

inline std::string toString(OpCode c) {
  switch (c) {
#define DEFINE_STR(op) \
  case OpCode::op:     \
    return #op;
    FORALL_OPCODES(DEFINE_STR)
#undef DEFINE_STR
  }
}

// TODO compare ops need to be in a more universally accessible place
enum class CompareOp {
  LessThan = 0,
  LessThanEqual = 1,
  Equal = 2,
  NotEqual = 3,
  GreaterThan = 4,
  GreaterThanEqual = 5,
  Is = 6,
  IsNot = 7,
};

struct Instruction {
  OpCode op;
  size_t arg1;
  Instruction(OpCode op_, int32_t arg1_) : op(op_), arg1(arg1_) {}

  friend std::ostream& operator<<(
      std::ostream& os,
      const Instruction& instruction);
};

} // namespace dynamic
} // namespace jit
} // namespace torch
