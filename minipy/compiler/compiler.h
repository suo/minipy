#pragma once

#include "minipy/jitparse/tree_views.h"
#include "minipy/interpreter/types.h"

#include <memory>

namespace minipy {

c10::intrusive_ptr<CodeObject> emit(Mod mod);
c10::intrusive_ptr<CodeObject> emit(Def def);
// TODO move this somewhere else
std::vector<std::string> gatherGlobals(const CodeObject& codeObject);
std::vector<std::string> gatherInstanceAttributes(const CodeObject& codeObject);
} // namespace dynamic
