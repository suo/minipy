#include "Obj.h"

#include <fmt/format.h>
#include <minipy/types.h>

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
Obj Dynamic::richCompare(Obj other, int opid) {
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
Obj Dynamic::str() const {
  throw std::runtime_error(
      fmt::format("str not implemented on type: '{}'", typeName_));
}

Obj Obj::call(Obj args) {
  if (!isDynamic()) {
    throw std::runtime_error("no getattr");
  }
  return toDynamic()->call(args);
}

Obj Obj::richCompare(Obj other, int opId) {
  switch (tag_) {
    case Tag::OBJECT:
      return toDynamicRef().richCompare(other, opId);
    case Tag::INT: {
      const auto lhs = toInt();
      // TODO this will crash
      const auto& rhs = other.toInt();
      // TODO don't use an int here
      switch (opId) {
        case 0: // lt
          return lhs < rhs;
        case 1: // lte
          return lhs <= rhs;
        case 2: // e
          return lhs == rhs;
        case 3: // ne
          return lhs != rhs;
        case 4: // gt
          return lhs > rhs;
        case 5: // gte
          return lhs >= rhs;
      }
    }
    case Tag::STRING: {
      const auto& lhs = toStringRef();
      // TODO this will crash
      const auto& rhs = other.toStringRef();
      // TODO don't use an int here
      switch (opId) {
        case 0: // lt
          return lhs < rhs;
        case 1: // lte
          return lhs <= rhs;
        case 2: // e
          return lhs == rhs;
        case 3: // ne
          return lhs != rhs;
        case 4: // gt
          return lhs > rhs;
        case 5: // gte
          return lhs >= rhs;
      }
    }
    default:
      throw std::runtime_error(
          fmt::format("comparisons not implemented on type: '{}'", typeName()));
  }
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

bool Obj::is(const Obj& rhs) const {
  const Obj& lhs = *this;
  if (lhs.isPtr()) {
    return rhs.isPtr() && lhs.tag_ == rhs.tag_ &&
        lhs.payload_.as_intrusive_ptr == rhs.payload_.as_intrusive_ptr;
  }
  return lhs == rhs;
}

bool operator!=(const Obj& lhs, const Obj& rhs) {
  return !(lhs == rhs);
}

const std::string& Obj::typeName() const {
  switch (tag_) {
    case Tag::NONE: {
      static const std::string ret = "None";
      return ret;
    }
    case Tag::INT: {
      static const std::string ret = "int";
      return ret;
    }
    case Tag::DOUBLE: {
      static const std::string ret = "double";
      return ret;
    }
    case Tag::BOOL: {
      static const std::string ret = "bool";
      return ret;
    }
    case Tag::STRING: {
      static const std::string ret = "str";
      return ret;
    }
    case Tag::OBJECT: {
      return toDynamic()->typeName_;
    }
  }
}

bool operator==(const Obj& lhs, const Obj& rhs) {
  switch (lhs.tag_) {
    case Obj::Tag::DOUBLE:
      return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
    case Obj::Tag::INT:
      return rhs.isInt() && lhs.toInt() == rhs.toInt();
    case Obj::Tag::BOOL:
      return rhs.isBool() && lhs.toBool() == rhs.toBool();
    case Obj::Tag::NONE:
      return rhs.isNone();
    case Obj::Tag::STRING:
      return lhs.toStringRef() == rhs.toStringRef();
    case Obj::Tag::OBJECT:
      return lhs.is(rhs);
  }
}

Obj::Obj(std::string s) : tag_(Tag::STRING) {
  auto ret = c10::make_intrusive<String>(std::move(s));
  payload_.as_intrusive_ptr = ret.release();
}

const std::string& Obj::toStringRef() const {
  if (tag_ != Tag::STRING) {
    throw std::runtime_error(
        fmt::format("Expected string, got {}", typeName()));
  }
  return static_cast<String*>(payload_.as_intrusive_ptr)->value();
}

Obj Obj::str() const {
  switch (tag_) {
    case Tag::NONE: {
      static const Obj ret = "None";
      return ret;
    }
    case Tag::INT:
      return Obj(fmt::format("{}", toInt()));
    case Tag::DOUBLE:
      return Obj(fmt::format("{}", toDouble()));
    case Tag::BOOL:
      return Obj(fmt::format("{}", toBool()));
    case Tag::STRING:
      return *this;
    case Tag::OBJECT:
      return toDynamic()->str();
  }
}

} // namespace dynamic
} // namespace jit
} // namespace torch
