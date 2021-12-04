// #include <minipy/interpreter.h>

// #include <fmt/format.h>
// #include <fmt/ostream.h>

// namespace torch {
// namespace jit {
// namespace dynamic {
// IValue Interpreter::run() {
//   fmt::print("\nRUNNING INTERPRETER\n\n");
//   while (true) {
//     Instruction instruction = code_.instructions[pc];
//     fmt::print("Running: {}\n", toString(instruction.op));
//     switch (instruction.op) {
//       case OpCode::LOAD_FAST: {
//         push(stack, frame_.fastLocals.at(instruction.arg1));
//         ++pc;
//       } break;
//       case OpCode::LOAD_GLOBAL: {
//         const std::string& name = code_.names[instruction.arg1];
//         auto it = frame_.globals.find(name);
//         if (it == frame_.globals.end()) {
//           throw std::runtime_error(
//               fmt::format("Couldn't find global '{}'", name));
//         }
//         push(stack, frame_.globals[name]);
//         ++pc;
//       } break;
//       case OpCode::LOAD_NAME: {
//         const std::string& name = code_.names[instruction.arg1];
//         auto it = frame_.locals.find(name);
//         if (it != frame_.locals.end()) {
//           push(stack, it->second);
//           ++pc;
//           break;
//         }
//         it = frame_.globals.find(name);
//         if (it != frame_.globals.end()) {
//           push(stack, it->second);
//           ++pc;
//           break;
//         }
//         TORCH_CHECK(false, fmt::format("Couldn't find name: '{}'", name));
//         ++pc;
//       } break;
//       case OpCode::STORE_NAME: {
//         IValue v = pop(stack);
//         const std::string& name = code_.names[instruction.arg1];
//         frame_.locals[name] = v;
//         ++pc;
//       } break;
//       case OpCode::LOAD_ATTR: {
//         IValue obj = pop(stack);
//         const std::string& name = code_.names[instruction.arg1];
//         push(stack, obj.getattr(name));
//         ++pc;
//       } break;
//       case OpCode::STORE_ATTR: {
//         IValue obj = pop(stack);
//         IValue valueToStore = pop(stack);
//         const std::string& name = code_.names[instruction.arg1];
//         obj.setattr(name, valueToStore);
//         ++pc;
//       } break;
//       case OpCode::LOAD_CONST: {
//         stack.emplace_back(code_.constants[instruction.arg1]);
//         ++pc;
//       } break;
//       case OpCode::CALL_FUNCTION: {
//         size_t numArgs = instruction.arg1;
//         Stack args = pop(stack, numArgs);
//         IValue callable = pop(stack);
//         IValue packedArgs = ivalue::Tuple::create(args);
//         IValue ret = callable.call(packedArgs);
//         push(stack, std::move(ret));
//         ++pc;
//       } break;
//       case OpCode::STORE_FAST: {
//         IValue obj = pop(stack);
//         frame_.fastLocals[instruction.arg1] = obj;
//         ++pc;
//       } break;
//       case OpCode::MAKE_FUNCTION: {
//         IValue name = pop(stack);
//         IValue co = pop(stack);
//         // Create a new function and push it on the stack
//         IValue func = c10::make_intrusive<FunctionObj>(
//             name.toStringRef(),
//             frame_.globals,
//             co.to<c10::intrusive_ptr<CodeObject>>());
//         push(stack, func);
//         ++pc;
//       } break;
//       case OpCode::LOAD_BUILD_CLASS: {
//         IValue buildClass = c10::make_intrusive<BuildClass>();
//         push(stack, buildClass);
//         ++pc;
//       } break;
//       case OpCode::COMPARE_OP: {
//         size_t comparisonOp = instruction.arg1;
//         IValue rhs = pop(stack);
//         IValue lhs = pop(stack);
//         IValue ret = lhs.richCompare(lhs, rhs, comparisonOp);
//         push(stack, std::move(ret));
//         ++pc;
//       } break;
//       case OpCode::POP_JUMP_IF_FALSE: {
//         size_t jumpTarget = instruction.arg1;
//         IValue v = pop(stack);
//         if (v.toBool() == false) {
//           pc = jumpTarget;
//         } else {
//           ++pc;
//         }
//       } break;
//       case OpCode::JUMP_ABSOLUTE: {
//         size_t jumpTarget = instruction.arg1;
//         pc = jumpTarget;
//       } break;
//       case OpCode::BINARY_ADD: {
//         IValue rhs = pop(stack);
//         IValue lhs = pop(stack);
//         TORCH_CHECK(rhs.isNumber());
//         push(stack, rhs.add(lhs));
//         ++pc;
//       } break;
//       case OpCode::POP_TOP: {
//         pop(stack);
//         ++pc;
//       } break;
//       case OpCode::RETURN_VALUE: {
//         return pop(stack);
//       } break;
//       default:
//         TORCH_CHECK(
//             false, fmt::format("UNHANDLED OP: {}\n",
//             toString(instruction.op)));
//     }
//   }
// }
// } // namespace dynamic
// } // namespace jit
// } // namespace torch
