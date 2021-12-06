#pragma once

#include <minipy/common/intrusive_ptr.h>
#include <string>
#include <string_view>

namespace torch {
namespace jit {
namespace dynamic {
namespace detail {
template <class T, class U>
c10::intrusive_ptr<T> static_intrusive_pointer_cast(c10::intrusive_ptr<U> r) {
  return c10::intrusive_ptr<T>::reclaim(static_cast<T*>(r.release()));
}
} // namespace detail

class Obj;

class Dynamic : public c10::intrusive_ptr_target {
 public:
  Dynamic(std::string typeName) : typeName_(std::move(typeName)) {}
  virtual ~Dynamic() {}

  virtual bool hasHasattr() const;
  virtual bool hasattr(const std::string& name) const;

  virtual bool hasGetattr() const;
  virtual Obj getattr(const std::string& name) const;

  virtual bool hasSetattr() const;
  virtual void setattr(const std::string& name, Obj value);

  virtual bool hasCall() const;
  virtual Obj call(Obj args);

  virtual bool hasRichCompare() const;
  virtual Obj richCompare(Obj o1, Obj o2, int opid);

  virtual bool isNumber() const;
  virtual Obj add(Obj other);

  std::string typeName_;
};

class Obj final {
 private:
  enum class Tag : uint32_t { NONE, INT, DOUBLE, BOOL, OBJECT };

 public:
  Obj() : Obj(nullptr) {}
  Obj(std::nullptr_t) : tag_(Tag::NONE) {}

  /// String support
  /*implicit*/ Obj(std::string v) { /*todo*/
  }
  /*implicit*/ Obj(std::string_view v) : Obj(std::string(v)) {}
  /*implicit*/ Obj(const char* v) : Obj(std::string(v)) {}

  bool isString() const {
    // return tag_ == Tag::STRING;
    return false;
  }
  const std::string& toStringRef() const;

  // generic v.to<at::Tensor>() implementations
  // that can be used in special functions like pop/push
  // that use template meta-programming.
  // prefer the directly named methods when you can,
  // since they are simpler to understand

  // Note: if you get linker errors saying one of these is missing,
  // change it to ... && = delete; and you will see better error messages for
  // why However, we cannot commit this because some compiler versions barf on
  // it.
  template <typename T>
  T to() &&;
  template <typename T>
  T to() const&;

  /*implicit*/ Obj(double v) : tag_(Tag::DOUBLE) {
    payload_.as_double = v;
  }
  /// Int
  /*implicit*/ Obj(int64_t v) : tag_(Tag::INT) {
    payload_.as_int = v;
  }

  int64_t toInt() const {
    if (tag_ != Tag::INT) {
      throw std::runtime_error("toInt() called on non-int Obj");
    }
    return payload_.as_int;
  }

  /// Bool support
  /*implicit*/ Obj(bool v) : tag_(Tag::BOOL) {
    payload_.as_bool = v;
  }
  bool isBool() const {
    return tag_ == Tag::BOOL;
  }
  bool toBool() const {
    return payload_.as_bool;
  }

  template <
      typename T,
      std::enable_if_t<std::is_base_of<Dynamic, T>::value, std::nullptr_t> =
          nullptr>
  /*implicit*/ Obj(c10::intrusive_ptr<T> v)
      : Obj(detail::static_intrusive_pointer_cast<Dynamic>(v)) {}
  /*implicit*/ Obj(c10::intrusive_ptr<Dynamic> v);

  bool isDynamic() const {
    return tag_ == Tag::OBJECT;
  }
  void clearToNone() {
    // TODO distinguish null and none, but I forgot why
    payload_.nul = nullptr;
    tag_ = Tag::NONE;
  }

  c10::intrusive_ptr<Dynamic> toDynamic() && {
    if (!isDynamic()) {
      throw std::runtime_error("toDynamicRef() called on non-dynamic Obj");
    }
    auto t = c10::intrusive_ptr<Dynamic>::reclaim(
        static_cast<Dynamic*>(payload_.as_intrusive_ptr));
    clearToNone();
    return t;
  }

  c10::intrusive_ptr<Dynamic> toDynamic() const& {
    if (!isDynamic()) {
      throw std::runtime_error("toDynamicRef() called on non-dynamic Obj");
    }
    auto r = c10::intrusive_ptr<Dynamic>::reclaim(
        static_cast<Dynamic*>(payload_.as_intrusive_ptr));
    auto p = r;
    r.release();
    return p;
  }

  const Dynamic& toDynamicRef() const {
    if (!isDynamic()) {
      throw std::runtime_error("toDynamicRef() called on non-dynamic Obj");
    }
    // AT_ASSERT(isDynamic(), "Expected Dynamic but got ", tagKind());
    return *static_cast<const Dynamic*>(payload_.as_intrusive_ptr);
  }

  Dynamic& toDynamicRef() {
    // AT_ASSERT(isDynamic(), "Expected Dynamic but got ", tagKind());
    if (!isDynamic()) {
      throw std::runtime_error("toDynamicRef() called on non-dynamic Obj");
    }
    return *static_cast<Dynamic*>(payload_.as_intrusive_ptr);
  }

  Obj richCompare(Obj other, int opId);

  Obj getattr(const std::string& name) {
    if (!isDynamic()) {
      throw std::runtime_error("no getattr");
    }

    return toDynamic()->getattr(name);
  }

  void setattr(const std::string& name, Obj v) {
    if (!isDynamic()) {
      throw std::runtime_error("no setattr");
    }

    toDynamic()->setattr(name, std::move(v));
  }

  // calling protocol
  // TODO implement iscallable
  Obj call(Obj args);

  // number protocol

  bool isNumber() const;
  Obj add(Obj other);

 private:
  union Payload {
    std::nullptr_t nul;
    int64_t as_int;
    double as_double;
    bool as_bool;
    c10::intrusive_ptr_target* as_intrusive_ptr;
  };

  Obj(Payload p, Tag t) : payload_(p), tag_(t) {}
  Payload payload_;
  Tag tag_;
  bool is_intrusive_ptr;
};

inline Obj::Obj(c10::intrusive_ptr<Dynamic> v) : tag_(Tag::OBJECT) {
  payload_.as_intrusive_ptr = v.release();
}

} // namespace dynamic
} // namespace jit
} // namespace torch
