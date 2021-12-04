#include "Obj.h"

#include <fmt/format.h>

namespace torch {
namespace jit {
namespace dynamic {

bool Dynamic::hasHasattr() const { return false; }
bool Dynamic::hasattr(const std::string &name) const {
  throw std::runtime_error(
      fmt::format("'hasattr' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasGetattr() const { return false; }
Obj Dynamic::getattr(const std::string &name) const {
  throw std::runtime_error(
      fmt::format("'getattr' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasSetattr() const { return false; }
void Dynamic::setattr(const std::string &name, Obj value) {
  throw std::runtime_error(
      fmt::format("'setattr' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasCall() const { return false; }
Obj Dynamic::call(Obj args) {
  throw std::runtime_error(
      fmt::format("'call' not implemented on type: '{}'", typeName_));
}
bool Dynamic::hasRichCompare() const { return false; }
Obj Dynamic::richCompare(Obj o1, Obj o2, int opid) {
  throw std::runtime_error(
      fmt::format("comparisons not implemented on type: '{}'", typeName_));
}
bool Dynamic::isNumber() const { return false; }
Obj Dynamic::add(Obj other) {
  throw std::runtime_error(
      fmt::format("add not implemented on type: '{}'", typeName_));
}
} // namespace dynamic
} // namespace jit
} // namespace torch
