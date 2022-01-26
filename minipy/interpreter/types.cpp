#include "types.h"

#include "minipy/interpreter/interpreter.h"

#include <sstream>

namespace minipy {
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

// AtenOp::AtenOp(const std::string& name)
//     : Dynamic("AtenOp"), name_(Symbol::fromQualString(name)) {}

// Obj AtenOp::call(Obj args) {
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

FunctionObj::FunctionObj(
    const std::string& name,
    std::unordered_map<std::string, Obj>& globals,
    c10::intrusive_ptr<CodeObject> co)
    : Dynamic("function"), globals_(globals), co_(std::move(co)) {}

Obj FunctionObj::call(Obj args) {
  // Set up stack frame
  FrameObject frame;
  // TODO eliminate the copy here
  frame.globals = globals_;
  frame.fastLocals.reserve(co_->varnames.size());
  // Add arguments to frame locals.
  for (const auto& arg : args.to<c10::intrusive_ptr<Tuple>>()->elements()) {
    frame.fastLocals.push_back(arg);
  }

  // Run an interpreter instance
  auto interpreter = Interpreter(*co_, frame);
  return interpreter.run();
}

BuildClass::BuildClass() : Dynamic("buildclass") {}

Obj BuildClass::call(Obj args_) {
  const auto& args = args_.to<c10::intrusive_ptr<Tuple>>()->elements();
  assert(args.size() == 2);
  const auto& name = args[1].toStringRef();
  const auto& classBuilder = args[0].to<c10::intrusive_ptr<FunctionObj>>();
  // Manually construct an interpreter instance, so that we can construct the
  // frame object appropriately.
  const auto& co = classBuilder->co_;
  FrameObject frame;
  frame.globals = classBuilder->globals_;
  auto interpreter = Interpreter(*co, frame).run();

  // The class builder function mostly acts to populate frame.locals as a
  // namespace dictionary.
  auto newClass = c10::make_intrusive<UserClass>(name, frame.locals);
  return newClass;
}

Obj UserClass::call(Obj args_) {
  // const auto& args = args_.toTuple()->elements();
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  auto intrusive_from_this = c10::intrusive_ptr<UserClass>::reclaim(this);

  // Allocate a new object
  auto obj = c10::make_intrusive<UserObject>(intrusive_from_this);

  // TODO call __init__ if it exists

  // For every function in the class namespace, make a bound method that lives
  // in the instance dict.  TODO this is not strictly correct; in Python this is
  // late-binding and governed by the descriptor protocol
  for (const auto& pr : namespace_) {
    const auto& name = pr.first;
    if (!pr.second.isDynamic()) {
      continue;
    }

    const auto& dyn = pr.second.toDynamic();
    auto fn = dynamic_intrusive_pointer_cast<FunctionObj>(dyn);
    if (!fn) {
      continue;
    }
    Obj boundMethod = c10::make_intrusive<MethodObj>(obj, std::move(fn));
    obj->setattr(name, std::move(boundMethod));
  }
  return obj;
}

void UserObject::setattr(const std::string& name, Obj value) {
  dict_[name] = std::move(value);
}

Obj UserObject::getattr(const std::string& name) const {
  // Generic getattr logic, (similar to PyObject_GenericGetAttr)

  // 1. Look it up in our instance variable dictionary
  auto it = dict_.find(name);
  if (it != dict_.end()) {
    return it->second;
  }

  // 2. Otherwise, look it up in the class namespace
  it = class_->namespace_.find(name);
  if (it != class_->namespace_.end()) {
    return it->second;
  }

  // 3. TODO look in base classes as well
  throw std::runtime_error(
      class_->typeName_ + "object has no attribute" + name);
}

Obj MethodObj::call(Obj args) {
  // Add self to the front of the arguments
  // TODO this is not very efficient, maybe there's some something in
  // vectorcall
  // protocol that would make this faster.
  const auto& args_ = args.to<c10::intrusive_ptr<Tuple>>()->elements();
  std::vector<Obj> argsWithSelf = {self_};
  for (const auto& arg : args_) {
    argsWithSelf.push_back(arg);
  }
  Obj argsWithSelfTuple = c10::make_intrusive<Tuple>(std::move(argsWithSelf));
  return fn_->call(argsWithSelfTuple);
}

// void ModuleObject::setattr(const std::string& name, Obj value) {
//   // TODO look at module.py setattr
//   attrs_.emplace(name, std::move(value));
// }

// Obj ModuleObject::getattr(const std::string& name) const {
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

// Obj ModuleObject::call(Obj args) {
//   auto forward = attrs_.at("forward");
//   return forward.call(std::move(args));
// }

// Obj DummyPyModule::getattr(const std::string& name) const {
//   auto it = attrs_.find(name);
//   if (it != attrs_.end()) {
//     return it->second;
//   }
//   throw std::runtime_error(
//       fmt::format("'{}' object has no attribute '{}'", typeName_, name));
// }

// void DummyPyModule::setattr(const std::string& name, Obj value) {
//   attrs_.emplace(name, std::move(value));
// }
namespace {
std::ostream& printList(
    std::ostream& out,
    const std::vector<Obj>& list,
    const std::string start,
    const std::string finish) {
  out << start;
  for (size_t i = 0; i < list.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << list[i].str().toStringRef();
  }
  out << finish;
  return out;
}

} // namespace

Obj Tuple::str() const {
  std::ostringstream ss;
  printList(ss, elements_, "(", ")");

  return Obj(ss.str());
}
} // namespace minipy
