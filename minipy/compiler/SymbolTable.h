#pragma once

#include "minipy/jitparse/tree_views.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace minipy {
enum class SymbolFlag : int {
  /* global stmt */
  DEF_GLOBAL = 1,
  /* assignment in code block */
  DEF_LOCAL = 2,
  /* formal parameter */
  DEF_PARAM = 2 << 1,
  /* nonlocal stmt */
  DEF_NONLOCAL = 2 << 2,
  /* name is used */
  USE = 2 << 3,
  /* name used but not defined in nested block */
  DEF_FREE = 2 << 4,
  /* free variable from class's method */
  DEF_FREE_CLASS = 2 << 5,
  /* assignment occurred via import */
  DEF_IMPORT = 2 << 6,
  /* this name is annotated */
  DEF_ANNOT = 2 << 7,
  /* this name is a comprehension iteration variable */
  DEF_COMP_ITER = 2 << 8
};

enum class SymbolScope { LOCAL, GLOBAL };

struct SymbolInfo {
  SymbolInfo() {}
  SymbolInfo& operator|=(SymbolFlag flag) {
    flags |= static_cast<int>(flag);
    return *this;
  }
  SymbolInfo& operator&=(SymbolFlag flag) {
    flags &= static_cast<int>(flag);
    return *this;
  }
  bool operator&(SymbolFlag flag) const {
    return flags & static_cast<int>(flag);
  }

  SymbolScope getScope() const;

 private:
  int flags = 0;
};

/**
 * class SymbolTableEntry
 *
 * Contains symbol information for a single scope (also called a "block" in
 * CPython).
 */
struct SymbolTableEntry {
  void dump(size_t indent = 0) const;
  enum class BlockKind { Module, Function, Class };
  BlockKind kind;
  // Name of this entry
  std::string name;
  // Mapping of identifier to symbol metadata.
  std::unordered_map<std::string, SymbolInfo> symbols;
  // If this is a function, the name of all the function arguments.
  std::vector<std::string> args;
  // For all blocks contained in this block (e.g. a method inside a class def).
  std::vector<SymbolTableEntry*> children;
  // Which AST node this SymbolTableEntry corresponds to.
  TreeRef ref;
};

class SymbolTable {
 public:
  static std::unique_ptr<SymbolTable> build(Mod module);
  static std::unique_ptr<SymbolTable> build(Def def);

  SymbolTableEntry* lookup(TreeRef astNode);

  void dump() const;

 private:
  // Private, use build()
  SymbolTable() {}
  class SymbolTableBuilder;
  std::vector<std::unique_ptr<SymbolTableEntry>> entries_;
  std::unordered_map<TreeRef, SymbolTableEntry*> lookupTable_;
};
} // namespace minipy
