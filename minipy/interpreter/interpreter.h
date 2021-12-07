#pragma once

#include "minipy/interpreter/types.h"
#include "minipy/interpreter/Stack.h"

namespace torch {
namespace jit {
namespace dynamic {

class Interpreter {
 public:
  Interpreter(const CodeObject& codeObject, FrameObject& frameObject)
      : code_(codeObject), frame_(frameObject) {}

  Obj run();

 private:
  Stack stack;
  size_t pc = 0;
  const CodeObject& code_;
  FrameObject& frame_;
};
} // namespace dynamic
} // namespace jit
} // namespace torch
