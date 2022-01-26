#pragma once

#include <string>
#include <string_view>
#include "minipy/common/intrusive_ptr.h"

namespace minipy {
template <class T, class U>
c10::intrusive_ptr<T> static_intrusive_pointer_cast(c10::intrusive_ptr<U> r) {
  return c10::intrusive_ptr<T>::reclaim(static_cast<T*>(r.release()));
}

template <class T, class U>
c10::intrusive_ptr<T> dynamic_intrusive_pointer_cast(c10::intrusive_ptr<U> r) {
  auto r2 = r.release();
  assert(r2);
  auto ret = dynamic_cast<T*>(r2);
  assert(ret);
  return c10::intrusive_ptr<T>::reclaim(ret);
}

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
  virtual Obj richCompare(Obj other, int opid);

  virtual bool isNumber() const;
  virtual Obj add(Obj other);

  virtual Obj str() const;

  std::string typeName_;
};

class Obj final {
 private:
  enum class Tag : uint32_t { NONE, INT, DOUBLE, BOOL, STRING, OBJECT };

 public:
  Obj() : Obj(nullptr) {}
  Obj(std::nullptr_t) : tag_(Tag::NONE) {}

  ~Obj() {
    if (isPtr()) {
      c10::raw::intrusive_ptr::decref(payload_.as_intrusive_ptr);
    }
  }
  Obj(const Obj& other) : Obj(other.payload_, other.tag_) {
    if (isPtr()) {
      c10::raw::intrusive_ptr::incref(payload_.as_intrusive_ptr);
    }
  }
  Obj& operator=(Obj&& rhs) noexcept {
    Obj(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  Obj& operator=(Obj const& rhs) {
    Obj(rhs).swap(*this);
    return *this;
  }

  void swap(Obj& rhs) noexcept {
    std::swap(payload_, rhs.payload_);
    std::swap(tag_, rhs.tag_);
  }

  /// String support
  // These are for covenience I guess?
  // TODO figure out whether these should just be modeled as dynamic objects or
  // deserve first-class support in Obj

  /*implicit*/ Obj(std::string v);
  // /*implicit*/ Obj(std::string_view v) : Obj(std::string(v)) {}
  // /*implicit*/ Obj(const char* v) : Obj(std::string(v)) {}

  bool isString() const {
    return tag_ == Tag::STRING;
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

  // TODO how does Py do this?
  const std::string& typeName() const;

  /**
   * Identity comparison. Checks if `this` is the same object as `rhs`. The
   * semantics are the same as Python's `is` operator.
   *
   * NOTE: Like in Python, this operation is poorly defined for primitive types
   * like numbers and strings. Prefer to use `==` unless you really want to
   * check identity equality.
   */
  bool is(const Obj& rhs) const;

  /**
   * This implements the same semantics as `bool(lhs == rhs)` in Python. which
   * is the same as `equals()` except for Tensor types.
   * TODO reword this, explain bool thing, remove Tensor comment
   */
  friend bool operator==(const Obj& lhs, const Obj& rhs);
  friend bool operator!=(const Obj& lhs, const Obj& rhs);

  /// None
  bool isNone() const {
    return tag_ == Tag::NONE;
  }

  /// Double
  /*implicit*/ Obj(double v) : tag_(Tag::DOUBLE) {
    payload_.as_double = v;
  }
  bool isDouble() const {
    return tag_ == Tag::DOUBLE;
  }
  bool toDouble() const {
    if (tag_ != Tag::DOUBLE) {
      throw std::runtime_error("toDouble() called on non-int Obj");
    }
    return payload_.as_double;
  }

  /// Int
  /*implicit*/ Obj(int64_t v) : tag_(Tag::INT) {
    payload_.as_int = v;
  }
  bool isInt() const {
    return tag_ == Tag::INT;
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
      : Obj(static_intrusive_pointer_cast<Dynamic>(v)) {}
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
    return moveToIntrusivePtr<Dynamic>();
  }

  c10::intrusive_ptr<Dynamic> toDynamic() const& {
    if (!isDynamic()) {
      throw std::runtime_error("toDynamicRef() called on non-dynamic Obj");
    }
    return toIntrusivePtr<Dynamic>();
  }

  template <class T>
  c10::intrusive_ptr<T> moveToIntrusivePtr() {
    auto t = c10::intrusive_ptr<T>::reclaim(
        static_cast<T*>(payload_.as_intrusive_ptr));
    clearToNone();
    return t;
  }

  template <typename T>
  c10::intrusive_ptr<T> toIntrusivePtr() const {
    auto r = c10::intrusive_ptr<T>::reclaim(
        static_cast<T*>(payload_.as_intrusive_ptr));
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

  // Object protocol
  Obj str() const;
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
  bool isPtr() const {
    return tag_ == Tag::OBJECT || tag_ == Tag::STRING;
  }

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
};

inline Obj::Obj(c10::intrusive_ptr<Dynamic> v) : tag_(Tag::OBJECT) {
  payload_.as_intrusive_ptr = v.release();
}

namespace detail {

template <class T>
struct _fake_type {};

template <
    typename T,
    std::enable_if_t<std::is_base_of<Dynamic, T>::value, std::nullptr_t> =
        nullptr>
c10::intrusive_ptr<T> generic_to(Obj obj, _fake_type<c10::intrusive_ptr<T>>) {
  auto ret = dynamic_intrusive_pointer_cast<T>(obj.toDynamic());
  return ret;
}

} // namespace detail

template <typename T>
inline T Obj::to() && {
  return generic_to(std::move(*this), detail::_fake_type<T>{});
}

template <typename T>
inline T Obj::to() const& {
  return generic_to(*this, detail::_fake_type<T>{});
}

} // namespace minipy
