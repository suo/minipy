#pragma once

#include <minipy/Obj.h>
#include <minipy/instruction.h>

#include <functional>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace dynamic {

// This leaks memory without a gc

// Instructions + tables that instructions reference.
struct CodeObject : Dynamic {
  CodeObject() : Dynamic("CodeObject") {}
  void dump() const;

  std::vector<std::string> names;
  std::vector<std::string> varnames;
  std::vector<Obj> constants;
  std::vector<Instruction> instructions;
};

// class AtenOp : public Dynamic {
//  public:
//   explicit AtenOp(const std::string& name);

//   Obj call(Obj args) override;
//   bool hasCall() const override {
//     return true;
//   }

//  private:
//   c10::Symbol name_;
// };

// class TorchModule : public Dynamic {
//  public:
//   TorchModule() : Dynamic("TorchModule") {}

//   Obj getattr(const std::string& name) const override {
//     return std::make_shared<AtenOp>(name);
//   }
//   bool hasGetattr() const override {
//     return true;
//   }
// };

class FunctionObj : public Dynamic {
 public:
  FunctionObj(
      const std::string& name,
      std::unordered_map<std::string, Obj>& globals,
      c10::intrusive_ptr<CodeObject> co);

  Obj call(Obj args) override;
  bool hasCall() const override {
    return true;
  }

  // TODO By copy! this is wrong
  std::unordered_map<std::string, Obj> globals_;
  c10::intrusive_ptr<CodeObject> co_;
};

class MethodObj : public Dynamic {
 public:
  MethodObj(Obj self, c10::intrusive_ptr<FunctionObj> fn)
      : Dynamic(fn->typeName_), self_(std::move(self)), fn_(std::move(fn)) {}

  Obj call(Obj args) override;
  bool hasCall() const override {
    return true;
  }

 private:
  Obj self_;
  c10::intrusive_ptr<FunctionObj> fn_;
};

class BuiltinFunction : public Dynamic {
 public:
  BuiltinFunction(const std::string& name, std::function<Obj(Obj)> callable)
      : Dynamic(name), callable_(std::move(callable)) {}

  Obj call(Obj args) override {
    return callable_(args);
  }
  bool hasCall() const override {
    return true;
  }

 private:
  std::function<Obj(Obj)> callable_;
};

struct FrameObject {
  std::unordered_map<std::string, Obj> globals;
  std::vector<Obj> fastLocals;
  std::unordered_map<std::string, Obj> locals;
};

class BuildClass : public Dynamic {
 public:
  BuildClass();
  Obj call(Obj args) override;
  bool hasCall() const override {
    return true;
  }
};

class UserClass : public Dynamic {
 public:
  UserClass(const std::string& name, std::unordered_map<std::string, Obj> ns)
      : Dynamic(name), namespace_(std::move(ns)) {}

  // bool hasattr(const std::string& name) const override;
  // Obj getattr(const std::string& name) const override;
  // void setattr(const std::string& name, Obj value) override;
  Obj call(Obj args) override;
  bool hasCall() const override {
    return true;
  }

  std::unordered_map<std::string, Obj> namespace_;
};

class UserObject : public Dynamic {
 public:
  UserObject(c10::intrusive_ptr<UserClass> cls)
      : Dynamic("obj"), class_(std::move(cls)) {}

  // bool hasattr(const std::string& name) const override;
  Obj getattr(const std::string& name) const override;
  bool hasGetattr() const override {
    return true;
  }
  void setattr(const std::string& name, Obj value) override;
  bool hasSetattr() const override {
    return true;
  }
  // Obj call(Obj args) override;

 private:
  std::unordered_map<std::string, Obj> dict_;
  c10::intrusive_ptr<UserClass> class_;
};

// represents an nn.Module
class ModuleObject : public Dynamic {
 public:
  ModuleObject() : Dynamic("module") {}

  Obj getattr(const std::string& name) const override;
  bool hasGetattr() const override {
    return true;
  }
  void setattr(const std::string& name, Obj value) override;
  bool hasSetattr() const override {
    return true;
  }
  Obj call(Obj value) override;
  bool hasCall() const override {
    return true;
  }

  //  private:
  std::unordered_map<std::string, Obj> parameters_;
  std::unordered_map<std::string, Obj> buffers_;
  std::unordered_map<std::string, Obj> modules_;
  std::unordered_map<std::string, Obj> attrs_;
  bool training_ = true;
};

class DummyPyModule : public Dynamic {
 public:
  DummyPyModule(std::string name) : Dynamic(std::move(name)) {}

  Obj getattr(const std::string& name) const override;
  bool hasGetattr() const override {
    return true;
  }
  void setattr(const std::string& name, Obj value) override;
  bool hasSetattr() const override {
    return true;
  }

 private:
  std::unordered_map<std::string, Obj> attrs_;
};

} // namespace dynamic
} // namespace jit
} // namespace torch
