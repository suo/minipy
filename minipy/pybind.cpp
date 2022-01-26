// #include <pybind11/pyTypes.h>
// #include "minipy/pybind.h"

// #include <fmt/format.h>
// #include "minipy/Compiler.h"
// #include "ATen/core/ivalue_inl.h"
// #include "jit/dynamic/Types.h"
// #include "jit/python/pybind_utils.h"

// namespace torch {
// namespace jit {
// namespace dynamic {

// // TODO we need an actual place where we initialize builtins and stuff
// auto len =
//     c10::make_intrusive<BuiltinFunction>("len", [](IValue args) -> IValue {
//       throw std::runtime_error("len not yet implemented");
//     });
// auto float_ =
//     c10::make_intrusive<BuiltinFunction>("float", [](IValue args) -> IValue {
//       throw std::runtime_error("float not yet implemented");
//     });

// void initJitDynamicBindings(PyObject* module) {
//   auto m = py::handle(module).cast<py::module>();

//   m.def("emit_bytecode", [](Def def) { return IValueHolder(emit(def)); });
//   m.def("gather_globals", [](const IValueHolder& obj) {
//     const auto& co =
//         dynamic_intrusive_pointer_cast<CodeObject>(obj.me_.toDynamic());
//     TORCH_INTERNAL_ASSERT(co);
//     return gatherGlobals(*co);
//   });
//   m.def("gather_instance_attributes", [](const IValueHolder& obj) {
//     const auto& co =
//         dynamic_intrusive_pointer_cast<CodeObject>(obj.me_.toDynamic());
//     TORCH_INTERNAL_ASSERT(co);
//     return gatherInstanceAttributes(*co);
//   });
//   m.def("get_builtin", [](const std::string& name) {
//     if (name == "len")  {
//       return IValueHolder(IValue(len));
//     } else if (name == "float") {
//       return IValueHolder(IValue(float_));
//     }
//     throw std::runtime_error("builtin not found");
//   });
//   m.def(
//       "build_function",
//       [](const std::string& name,
//          const IValueHolder& obj,
//          const std::unordered_map<std::string, IValueHolder> globalDict) {
//         auto codeObject =
//             dynamic_intrusive_pointer_cast<CodeObject>(obj.me_.toDynamic());
//         std::unordered_map<std::string, IValue> globals;
//         for (const auto& pr : globalDict) {
//           globals[pr.first] = pr.second.me_;
//         }
//         return IValueHolder(c10::make_intrusive<FunctionObj>(
//             name, globals, std::move(codeObject)));
//       });
//   m.def("try_to_ivalue", [](const py::handle obj) {
//     return IValueHolder(toIValue(obj, tryToInferType(obj).type()));
//   });

//   py::class_<IValueHolder>(m, "IValue")
//       .def(
//           "__call__",
//           [](IValueHolder& self, py::args args) {
//             std::vector<IValue> ivalueArgs;
//             for (const auto& arg : args) {
//               ivalueArgs.push_back(toIValue(arg, tryToInferType(arg).type()));
//             }
//             IValue argTuple = c10::ivalue::Tuple::create(std::move(ivalueArgs));
//             return self.me_.call(std::move(argTuple));
//           })
//       .def(
//           "__setattr__",
//           [](IValueHolder& self, const std::string& name, const IValueHolder& value) {
//             self.me_.setattr(name, value.me_);
//           })
//       .def("__getattr__", [](IValueHolder& self, const std::string& name) {
//         return self.me_.getattr(name);
//       });

//   // TODO how to reduce duplication between IValueHolder and this?
//   py::class_<DynamicModuleHolder>(m, "DynamicModule")
//       // TODO kwargs
//       .def(
//           "__call__",
//           [](DynamicModuleHolder& self, py::args args) {
//             HANDLE_TH_ERRORS
//             // Prepare the arguments into a stack
//             std::vector<IValue> ivalueArgs;
//             for (const auto& arg : args) {
//               ivalueArgs.push_back(toIValue(arg, tryToInferType(arg).type()));
//             }
//             IValue argTuple = c10::ivalue::Tuple::create(std::move(ivalueArgs));
//             return self.me_->call(std::move(argTuple));
//             END_HANDLE_TH_ERRORS_PYBIND
//           })
//       .def(
//           "__setattr__",
//           [](DynamicModuleHolder& self,
//              const std::string& name,
//              py::handle value) {
//             auto type = tryToInferType(value);
//             self.me_->setattr(name, toIValue(value, type.type()));
//           })
//       .def(
//           "__getattr__",
//           [](DynamicModuleHolder& self, const std::string& name) {
//             return self.me_->getattr(name);
//           })
//       // .def(
//       //     "define_method",
//       //     [](DynamicModuleHolder& self, const std::string& name, Def def) {
//       //       auto codeObject = emit(def);
//       //       codeObject->dump();
//       //       // const auto globalNames = gatherGlobals(*codeObject);
//       //       // TODO globals?????
//       //       std::unordered_map<std::string, IValue> globals;
//       //       auto fn = c10::make_intrusive<FunctionObj>(
//       //           def.name().name(), globals, std::move(codeObject));
//       //       IValue method = c10::make_intrusive<MethodObj>(
//       //           self.me_, std::move(fn));
//       //       self.me_->attrs_.emplace(name, std::move(method));
//       //       // return globalNames;
//       //     })
//       .def(
//           "register_method",
//           [](DynamicModuleHolder& self,
//              const std::string& name,
//              const IValueHolder& value) {
//             auto fn = dynamic_intrusive_pointer_cast<FunctionObj>(
//                 value.me_.toDynamic());
//             IValue method =
//                 c10::make_intrusive<MethodObj>(self.me_, std::move(fn));
//             self.me_->attrs_.emplace(name, std::move(method));
//           })
//       .def(
//           "register_attribute",
//           [](DynamicModuleHolder& self,
//              const std::string& name,
//              const IValueHolder& value) {
//             self.me_->attrs_.emplace(name, value.me_);
//           })
//       .def(
//           "register_parameter",
//           [](DynamicModuleHolder& self,
//              const std::string& name,
//              py::handle value) {
//             auto type = tryToInferType(value).type();
//             if (!type->isSubtypeOf(NoneType::get()) &&
//                 !type->isSubtypeOf(TensorType::get())) {
//               throw std::runtime_error(
//                   "wrong type for parameter, must be None or Tensor");
//             }
//             self.me_->parameters_.emplace(name, toIValue(value, type));
//           })
//       .def(
//           "register_module",
//           [](DynamicModuleHolder& self,
//              const std::string& name,
//              py::handle value) {
//             auto submodule = py::cast<DynamicModuleHolder>(value);
//             self.me_->modules_.emplace(name, submodule.me_);
//           })
//       .def(
//           "register_buffer",
//           [](DynamicModuleHolder& self,
//              const std::string& name,
//              py::handle value) {
//             auto type = tryToInferType(value).type();
//             if (!type->isSubtypeOf(NoneType::get()) &&
//                 !type->isSubtypeOf(TensorType::get())) {
//               throw std::runtime_error(
//                   "wrong type for buffer, must be None or Tensor");
//             }
//             self.me_->buffers_.emplace(name, toIValue(value, type));
//           });

//   m.def("make_dynamic_module", []() {
//     return DynamicModuleHolder(c10::make_intrusive<ModuleObject>());
//   });
//   m.def("make_dummy_py_module", [](std::string name) {
//     return IValueHolder(
//         IValue(c10::make_intrusive<DummyPyModule>(std::move(name))));
//   });
//   m.def("make_aten_builtin", [](std::string name) {
//     return IValueHolder(IValue(c10::make_intrusive<AtenOp>(std::move(name))));
//   });
// }
// } // namespace dynamic
// } // namespace jit
// } // namespace torch
