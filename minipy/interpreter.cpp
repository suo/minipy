#include <minipy/interpreter.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace torch {
namespace jit {
namespace dynamic {
Obj Interpreter::run() {
  fmt::print("\nRUNNING INTERPRETER\n\n");
  while (true) {
    Instruction instruction = code_.instructions[pc];
    fmt::print("Running: {}\n", toString(instruction.op));
    switch (instruction.op) {
      case OpCode::LOAD_FAST: {
        push(stack, frame_.fastLocals.at(instruction.arg1));
        ++pc;
      } break;
      case OpCode::LOAD_GLOBAL: {
        const std::string& name = code_.names[instruction.arg1];
        auto it = frame_.globals.find(name);
        if (it == frame_.globals.end()) {
          throw std::runtime_error(
              fmt::format("Couldn't find global '{}'", name));
        }
        push(stack, frame_.globals[name]);
        ++pc;
      } break;
      case OpCode::LOAD_NAME: {
        const std::string& name = code_.names[instruction.arg1];
        auto it = frame_.locals.find(name);
        if (it != frame_.locals.end()) {
          push(stack, it->second);
          ++pc;
          break;
        }
        it = frame_.globals.find(name);
        if (it != frame_.globals.end()) {
          push(stack, it->second);
          ++pc;
          break;
        }
        assert(false);
        //  fmt::format("Couldn't find name: '{}'", name));
        ++pc;
      } break;
      case OpCode::STORE_NAME: {
        Obj v = pop(stack);
        const std::string& name = code_.names[instruction.arg1];
        frame_.locals[name] = v;
        ++pc;
      } break;
      case OpCode::LOAD_ATTR: {
        Obj obj = pop(stack);
        const std::string& name = code_.names[instruction.arg1];
        push(stack, obj.getattr(name));
        ++pc;
      } break;
      case OpCode::STORE_ATTR: {
        Obj obj = pop(stack);
        Obj valueToStore = pop(stack);
        const std::string& name = code_.names[instruction.arg1];
        obj.setattr(name, valueToStore);
        ++pc;
      } break;
      case OpCode::LOAD_CONST: {
        stack.emplace_back(code_.constants[instruction.arg1]);
        ++pc;
      } break;
      case OpCode::CALL_FUNCTION: {
        size_t numArgs = instruction.arg1;
        Stack args = pop(stack, numArgs);
        Obj callable = pop(stack);
        Obj packedArgs = c10::make_intrusive<Tuple>(args);
        Obj ret = callable.call(packedArgs);
        push(stack, std::move(ret));
        ++pc;
      } break;
      case OpCode::STORE_FAST: {
        Obj obj = pop(stack);
        frame_.fastLocals[instruction.arg1] = obj;
        ++pc;
      } break;
      case OpCode::MAKE_FUNCTION: {
        Obj name = pop(stack);
        Obj co = pop(stack);
        // Create a new function and push it on the stack
        Obj func = c10::make_intrusive<FunctionObj>(
            name.toStringRef(),
            frame_.globals,
            co.to<c10::intrusive_ptr<CodeObject>>());
        push(stack, func);
        ++pc;
      } break;
      case OpCode::LOAD_BUILD_CLASS: {
        Obj buildClass = c10::make_intrusive<BuildClass>();
        push(stack, buildClass);
        ++pc;
      } break;
      case OpCode::COMPARE_OP: {
        size_t comparisonOp = instruction.arg1;
        Obj rhs = pop(stack);
        Obj lhs = pop(stack);
        Obj ret = lhs.richCompare(rhs, comparisonOp);
        push(stack, std::move(ret));
        ++pc;
      } break;
      case OpCode::POP_JUMP_IF_FALSE: {
        size_t jumpTarget = instruction.arg1;
        Obj v = pop(stack);
        if (v.toBool() == false) {
          pc = jumpTarget;
        } else {
          ++pc;
        }
      } break;
      case OpCode::JUMP_ABSOLUTE: {
        size_t jumpTarget = instruction.arg1;
        pc = jumpTarget;
      } break;
      case OpCode::BINARY_ADD: {
        Obj rhs = pop(stack);
        Obj lhs = pop(stack);
        assert(rhs.isNumber());
        push(stack, rhs.add(lhs));
        ++pc;
      } break;
      case OpCode::POP_TOP: {
        pop(stack);
        ++pc;
      } break;
      case OpCode::RETURN_VALUE: {
        return pop(stack);
      } break;
      default:
        assert(
            false);
            // , fmt::format("UNHANDLED OP: {}\n",
            // toString(instruction.op)));
    }
  }
}
} // namespace dynamic
} // namespace jit
} // namespace torch
