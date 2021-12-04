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

class Dynamic;

class Obj final {
 private:
  enum class Tag : uint32_t { NONE, INT, DOUBLE, BOOL, OBJECT };

 public:
  Obj() : Obj(nullptr) {}
  Obj(std::nullptr_t) : tag_(Tag::NONE) {}

  // STRINGS!
  /*implicit*/ Obj(std::string v) { /*todo*/
  }
  /*implicit*/ Obj(std::string_view v) : Obj(std::string(v)) {}
  /*implicit*/ Obj(const char* v) : Obj(std::string(v)) {}

  /*implicit*/ Obj(double v) : tag_(Tag::DOUBLE) {
    payload_.as_double = v;
  }
  /*implicit*/ Obj(int64_t v) : tag_(Tag::INT) {
    payload_.as_int = v;
  }

  template <
      typename T,
      std::enable_if_t<std::is_base_of<Dynamic, T>::value, std::nullptr_t> =
          nullptr>
  /*implicit*/ Obj(c10::intrusive_ptr<T> v)
      : Obj(detail::static_intrusive_pointer_cast<Dynamic>(v)) {}
  /*implicit*/ Obj(c10::intrusive_ptr<Dynamic> v);

  // bool isDynamic() const {
  //   return tag == Tag::OBJECT;
  // }
  // c10::intrusive_ptr<Dynamic> toDynamic() &&;
  // c10::intrusive_ptr<Dynamic> toDynamic() const&;
  // const ivalue::Dynamic& toDynamicRef() const;
  // ivalue::Dynamic& toDynamicRef();

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

inline Obj::Obj(c10::intrusive_ptr<Dynamic> v) : tag_(Tag::OBJECT) {
  payload_.as_intrusive_ptr = v.release();
}
} // namespace dynamic
} // namespace jit
} // namespace torch
