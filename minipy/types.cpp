#include <minipy/types.h>

// #include <ATen/core/stack.h>
// #include <minipy/interpreter.h>
// #include <torch/csrc/jit/runtime/operator.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
// #include "ATen/core/ivalue_inl.h"

namespace torch {
namespace jit {
namespace dynamic {
// namespace {
// Operation resolveOverloads(Symbol name_, Stack& stack) {
//   const auto& overloads = getAllOperatorsFor(name_);
//   for (const auto& overload : overloads) {
//     const FunctionSchema& schema = overload->schema();
//     // Very dumb overload resolution!
//     if (schema.overload_name() == "Tensor") {
//       fmt::print("Calling schema: {}\n", schema);
//       for (const auto& arg : schema.arguments()) {
//         if (const auto& default_ = arg.default_value()) {
//           push(stack, default_);
//         }
//       }
//       return overload->getOperation();
//     }
//     if (name_ == aten::tensor && schema.overload_name() == "int") {
//       fmt::print("Calling schema: {}\n", schema);
//       for (const auto& arg : schema.arguments()) {
//         if (const auto& default_ = arg.default_value()) {
//           push(stack, default_);
//         }
//       }
//       return overload->getOperation();
//     }
//   }
//   TORCH_CHECK(
//       false,
//       fmt::format(
//           "Couldn't find correct overload for op: '{}'. Tried: {}",
//           name_.toQualString(),
//           fmap(overloads, [](const auto& overload) {
//             return overload->schema();
//           })));
// }
// } // namespace

void CodeObject::dump() const {
  size_t offset = 0;
  for (const Instruction& instruction : instructions) {
    if (instruction.op == OpCode::LOAD_FAST ||
        instruction.op == OpCode::STORE_FAST) {
      fmt::print(
          "{:<8}{} ({})\n",
          offset++,
          instruction,
          varnames.at(instruction.arg1));
    } else if (
        instruction.op == OpCode::STORE_GLOBAL ||
        instruction.op == OpCode::LOAD_GLOBAL ||
        instruction.op == OpCode::LOAD_METHOD ||
        instruction.op == OpCode::LOAD_NAME ||
        instruction.op == OpCode::STORE_NAME ||
        instruction.op == OpCode::LOAD_ATTR) {
      fmt::print(
          "{:<8}{} ({})\n", offset++, instruction, names.at(instruction.arg1));
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

// AtenOp::AtenOp(const std::string& name)
//     : ivalue::Dynamic("AtenOp"), name_(Symbol::fromQualString(name)) {}

// IValue AtenOp::call(IValue args) {
//   Stack stack = args.toTuple()->elements();
//   Operation op = resolveOverloads(name_, stack);
//   fmt::print("STACK:\n");
//   size_t i = 0;
//   for (const auto& v : stack) {
//     fmt::print("{}: \t{}\n", i++, v);
//   }
//   op(&stack);

//   return pop(stack);
// }

// FunctionObj::FunctionObj(
//     const std::string& name,
//     std::unordered_map<std::string, IValue>& globals,
//     c10::intrusive_ptr<CodeObject> co)
//     : ivalue::Dynamic("function"), globals_(globals), co_(std::move(co)) {}

// IValue FunctionObj::call(IValue args) {
//   // Set up stack frame
//   FrameObject frame;
//   // TODO eliminate the copy here
//   frame.globals = globals_;
//   frame.fastLocals.reserve(co_->varnames.size());
//   // Add arguments to frame locals.
//   for (const auto& arg : args.toTuple()->elements()) {
//     frame.fastLocals.push_back(arg);
//   }

//   // Run an interpreter instance
//   auto interpreter = Interpreter(*co_, frame);
//   return interpreter.run();
// }

// BuildClass::BuildClass() : ivalue::Dynamic("buildclass") {}

// IValue BuildClass::call(IValue args_) {
//   const auto& args = args_.toTuple()->elements();
//   TORCH_INTERNAL_ASSERT(args.size() == 2);
//   const auto& name = args[1].toStringRef();
//   const auto& classBuilder = args[0].to<c10::intrusive_ptr<FunctionObj>>();
//   // Manually construct an interpreter instance, so that we can construct the
//   // frame object appropriately.
//   const auto& co = classBuilder->co_;
//   FrameObject frame;
//   frame.globals = classBuilder->globals_;
//   auto interpreter = Interpreter(*co, frame).run();

//   // The class builder function mostly acts to populate frame.locals as a
//   // namespace dictionary.
//   auto newClass = c10::make_intrusive<UserClass>(name, frame.locals);
//   return newClass;
// }

// IValue UserClass::call(IValue args_) {
//   // const auto& args = args_.toTuple()->elements();
//   c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
//                                          // from a raw `this` pointer
//                                          // so we need to bump the refcount
//                                          // to account for this ownership
//   auto intrusive_from_this = c10::intrusive_ptr<UserClass>::reclaim(this);

//   // Allocate a new object
//   auto obj = c10::make_intrusive<UserObject>(intrusive_from_this);

//   // TODO call __init__ if it exists

//   // For every function in the class namespace, make a bound method that
//   lives
//   // in the instance dict.
//   // TODO this is not strictly correct; in Python this is late-binding and
//   // governed by the descriptor protocol
//   for (const auto& pr : namespace_) {
//     const auto& name = pr.first;
//     if (!pr.second.isDynamic()) {
//       continue;
//     }

//     const auto& dyn = pr.second.toDynamic();
//     auto fn = dynamic_intrusive_pointer_cast<FunctionObj>(dyn);
//     if (!fn) {
//       continue;
//     }
//     IValue boundMethod = c10::make_intrusive<MethodObj>(obj, std::move(fn));
//     obj->setattr(name, std::move(boundMethod));
//   }
//   return obj;
// }

// void UserObject::setattr(const std::string& name, IValue value) {
//   dict_[name] = std::move(value);
// }

// IValue UserObject::getattr(const std::string& name) const {
//   // Generic getattr logic, (similar to PyObject_GenericGetAttr)

//   // 1. Look it up in our instance variable dictionary
//   auto it = dict_.find(name);
//   if (it != dict_.end()) {
//     return it->second;
//   }

//   // 2. Otherwise, look it up in the class namespace
//   it = class_->namespace_.find(name);
//   if (it != class_->namespace_.end()) {
//     return it->second;
//   }

//   // 3. TODO look in base classes as well
//   throw std::runtime_error(fmt::format(
//       "'{}' object has no attribute '{}'", class_->typeName_, name));
// }

// IValue MethodObj::call(IValue args) {
//   // Add self to the front of the arguments
//   // TODO this is not very efficient, maybe there's some something in
//   vectorcall
//   // protocol that would make this faster.
//   const auto& args_ = args.toTuple()->elements();
//   std::vector<IValue> argsWithSelf = {self_};
//   for (const auto& arg : args_) {
//     argsWithSelf.push_back(arg);
//   }
//   IValue argsWithSelfTuple = ivalue::Tuple::create(std::move(argsWithSelf));
//   return fn_->call(argsWithSelfTuple);
// }

// void ModuleObject::setattr(const std::string& name, IValue value) {
//   // TODO look at module.py setattr
//   attrs_.emplace(name, std::move(value));
// }

// IValue ModuleObject::getattr(const std::string& name) const {
//   // Check dict (implicit behavior before we hit getattr)
//   // TODO perhaps this behavior should be uniform.
//   auto it = attrs_.find(name);
//   if (it != attrs_.end()) {
//     return it->second;
//   }
//   // Check parameters
//   it = parameters_.find(name);
//   if (it != parameters_.end()) {
//     return it->second;
//   }
//   // Check buffers
//   it = buffers_.find(name);
//   if (it != buffers_.end()) {
//     return it->second;
//   }
//   // Check modules
//   it = modules_.find(name);
//   if (it != modules_.end()) {
//     return it->second;
//   }
//   throw std::runtime_error(
//       fmt::format("'{}' object has no attribute '{}'", "nn.module", name));
// }

// IValue ModuleObject::call(IValue args) {
//   auto forward = attrs_.at("forward");
//   return forward.call(std::move(args));
// }

// IValue DummyPyModule::getattr(const std::string& name) const {
//   auto it = attrs_.find(name);
//   if (it != attrs_.end()) {
//     return it->second;
//   }
//   throw std::runtime_error(
//       fmt::format("'{}' object has no attribute '{}'", typeName_, name));
// }

// void DummyPyModule::setattr(const std::string& name, IValue value) {
//   attrs_.emplace(name, std::move(value));
// }
} // namespace dynamic
} // namespace jit
} // namespace torch