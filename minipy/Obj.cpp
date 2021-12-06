#include "Obj.h"

#include <fmt/format.h>

namespace torch {
namespace jit {
namespace dynamic {

bool Dynamic::hasHasattr() const {
  return false;
}
bool Dynamic::hasattr(const std::string& name) const {
  throw std::runtime_error(
      fmt::format("'hasattr' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasGetattr() const {
  return false;
}
Obj Dynamic::getattr(const std::string& name) const {
  throw std::runtime_error(
      fmt::format("'getattr' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasSetattr() const {
  return false;
}
void Dynamic::setattr(const std::string& name, Obj value) {
  throw std::runtime_error(
      fmt::format("'setattr' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasCall() const {
  return false;
}
Obj Dynamic::call(Obj args) {
  throw std::runtime_error(
      fmt::format("'call' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasRichCompare() const {
  return false;
}
Obj Dynamic::richCompare(Obj o1, Obj o2, int opid) {
  throw std::runtime_error(
      fmt::format("comparisons not implemented on type: '{}'", typeName_));
}
bool Dynamic::isNumber() const {
  return false;
}
Obj Dynamic::add(Obj other) {
  throw std::runtime_error(
      fmt::format("add not implemented on type: '{}'", typeName_));
}

Obj Obj::call(Obj args) {
  if (!isDynamic()) {
    throw std::runtime_error("no getattr");
  }
  return toDynamic()->call(args);
}

Obj Obj::richCompare(Obj other, int opId) {
  throw std::runtime_error("no richCompare");
}

// TODO figure out a better way of checking the numeric protocol
bool Obj::isNumber() const {
  switch (tag_) {
    case Tag::OBJECT:
      return toDynamicRef().isNumber();
    case Tag::INT:
    case Tag::DOUBLE:
      return true;
    default:
      return false;
  }
}

// TODO this is not correct, use the correct binary protocol
Obj Obj::add(Obj other) {
  switch (tag_) {
    case Tag::INT:
      return toInt() + other.toInt();
    default:
    throw std::runtime_error("no add");
  }
}

const std::string& Obj::toStringRef() const {
  // todo error
  if (!isString()) {
    throw std::runtime_error("Expected String but got TODO");
  }

  static std::string foo = "";
  return foo;
  // return static_cast<const c10::ivalue::ConstantString*>(
  //            payload.u.as_intrusive_ptr)
  //     ->string();
}

} // namespace dynamic
} // namespace jit
} // namespace torch
