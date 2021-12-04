// #pragma once

// #include <minipy/types.h>

// namespace torch {
// namespace jit {
// namespace dynamic {
// class Interpreter {
//  public:
//   Interpreter(const CodeObject& codeObject, FrameObject& frameObject)
//       : code_(codeObject), frame_(frameObject) {}

//   IValue run();

//  private:
//   Stack stack;
//   size_t pc = 0;
//   const CodeObject& code_;
//   FrameObject& frame_;
// };
// } // namespace dynamic
// } // namespace jit
// } // namespace torch
