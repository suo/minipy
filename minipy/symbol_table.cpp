#include <minipy/symbol_table.h>

#include "minipy/jitparse/lexer.h"
#include "minipy/jitparse/tree_views.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <cassert>
#include <unordered_set>

namespace torch {
namespace jit {
namespace dynamic {
SymbolScope SymbolInfo::getScope() const {
  if (*this & SymbolFlag::DEF_LOCAL || *this & SymbolFlag::DEF_PARAM) {
    return SymbolScope::LOCAL;
  } else {
    // TODO this is wrong, we can have stuff besides local and global
    return SymbolScope::GLOBAL;
  }
}

class SymbolTable::SymbolTableBuilder {
 public:
  explicit SymbolTableBuilder() {}

  std::unique_ptr<SymbolTable> build(Mod module) {
    // Can't use make_unique because SymbolTable's constructor is private
    table_ = std::unique_ptr<SymbolTable>(new SymbolTable());
    push(module);
    visit(module.body());
    pop();
    return std::move(table_);
  }

  std::unique_ptr<SymbolTable> build(Def def) {
    // Can't use make_unique because SymbolTable's constructor is private
    table_ = std::unique_ptr<SymbolTable>(new SymbolTable());
    push(def);
    visit(def.decl().params());
    visit(def.statements());
    pop();
    return std::move(table_);
  }

 private:
  template <typename T>
  void visit(List<T> elements) {
    for (T element : elements) {
      visit(element);
    }
  }

  void visit(Def def) {
    // Add a symbol to the outer block representing this def
    addSymbol(def.name().name(), SymbolFlag::DEF_LOCAL);

    // Then construct a new block and populate it.
    push(def);
    visit(def.decl().params());
    visit(def.statements());
    pop();
  }

  void visit(ClassDef classDef) {
    // Add a symbol to the outer scope representing this def
    addSymbol(classDef.name().name(), SymbolFlag::DEF_LOCAL);

    // Then construct a new block and populate it.
    push(classDef);
    visit(classDef.body());
    pop();
  }

  void visit(Param param) {
    addSymbol(param.ident().name(), SymbolFlag::DEF_PARAM);
  }

  void visit(BinOp binOp) {
    visit(binOp.lhs());
    visit(binOp.rhs());
  }

  void visit(UnaryOp unaryOp) {
    visit(unaryOp.operand());
  }

  void visit(Expr expr) {
    switch (expr.kind()) {
      case TK_AND:
      case TK_OR:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '-':
      case TK_IS:
      case TK_ISNOT:
      case '*':
      case '/':
      case '%':
      case '&':
      case '^':
      case '|':
      case '>':
      case '<':
      case '+':
      case TK_FLOOR_DIV: {
        return visit(BinOp(expr));
      } break;
      case TK_APPLY: {
        Apply apply(expr);
        visit(apply.callee());
        visit(apply.inputs());
        return;
      } break;
      case '.': {
        Select select(expr);
        visit(select.value());
      } break;
      case TK_VAR: {
        Var var(expr);
        if (isAssignmentContext_) {
          addSymbol(var.name().name(), SymbolFlag::DEF_LOCAL);
        } else {
          addSymbol(var.name().name(), SymbolFlag::USE);
        }
      } break;
      case TK_UNARY_MINUS:
      case '~':
      case TK_NOT: {
        visit(UnaryOp(expr));
      } break;
      case TK_CONST:
      case TK_LIST_LITERAL:
      case TK_TUPLE_LITERAL:
      case TK_DICT_LITERAL:
      case TK_STRINGLITERAL:
      case TK_NONE:
        // Nothing to do here
        break;
      case TK_IF_EXPR:
      case TK_STARRED:
      case TK_TRUE:
      case TK_FALSE:
      case TK_CAST:
      case TK_SUBSCRIPT:
      case TK_SLICE_EXPR:
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case TK_LIST_COMP:
      case TK_DOTS:
      case TK_IN:
      default:
        // throw ErrorReport(expr.range())
        //     << "symtable: unsupported expr " << kindToString(expr.kind());
        fmt::print(
            "symtable: unsupported expr {}\n", kindToString(expr.kind()));
    }
  }

  void visit(const Stmt& stmt) {
    switch (stmt.kind()) {
      case TK_ASSIGN: {
        Assign assign(stmt);
        {
          AssignmentContextGuard g(this);
          visit(assign.lhs());
        }
        visit(assign.rhs().get());
      } break;
      case TK_RETURN: {
        Return ret(stmt);
        visit(ret.expr());
      } break;
      case TK_DEF:
        visit(Def(stmt));
        break;
      case TK_CLASS_DEF:
        visit(ClassDef(stmt));
        break;
      case TK_IF: {
        auto if_ = If(stmt);
        visit(if_.cond());
        visit(if_.trueBranch());
        visit(if_.falseBranch());
      } break;
      case TK_FOR: {
        auto for_ = For(stmt);
        visit(for_.targets());
        visit(for_.itrs());
        visit(for_.body());
      } break;
      case TK_AUG_ASSIGN: {
        AugAssign augAssign(stmt);
        visit(augAssign.lhs());
        visit(augAssign.rhs());
      } break;
      case TK_EXPR_STMT: {
        visit(ExprStmt(stmt).expr());
      } break;
      case TK_RAISE: {
        // TODO support raises more
      } break;
      case TK_ASSERT: {
        Assert assert_(stmt);
        visit(assert_.test());
        if (assert_.msg().present()) {
          visit(assert_.msg().get());
        }
      } break;
      case TK_WHILE:
      case TK_GLOBAL:

      case TK_PASS:
      case TK_BREAK:
      case TK_DELETE:
      case TK_CONTINUE:
        throw ErrorReport(stmt.range())
            << "symtable: unsupported stmt " << kindToString(stmt.kind());
    }
  }

  void addSymbol(const std::string& name, SymbolFlag flag) {
    auto& symbols = cur()->symbols;
    auto it = symbols.find(name);
    if (flag == SymbolFlag::DEF_PARAM) {
      if (it != symbols.end()) {
        // TODO better error message
        throw std::runtime_error(
            "symtable: duplicate parameter name not supported:");
      }
      cur()->args.push_back(name);
    }
    symbols[name] |= flag;
  }

  // TODO derive kind from ref kind, this is because we have no module today
  void push(TreeRef ref) {
    auto prev = curEntry_;
    table_->entries_.push_back(std::make_unique<SymbolTableEntry>());
    curEntry_ = table_->entries_.back().get();
    switch (ref->kind()) {
      case TK_MODULE:
        curEntry_->kind = SymbolTableEntry::BlockKind::Module;
        curEntry_->name = "Module";
        break;
      case TK_CLASS_DEF:
        curEntry_->kind = SymbolTableEntry::BlockKind::Class;
        curEntry_->name = ClassDef(ref).name().name();
        break;
      case TK_DEF:
        curEntry_->kind = SymbolTableEntry::BlockKind::Function;
        curEntry_->name = Def(ref).name().name();
        break;
      default:
        assert(false);
    }

    curEntry_->ref = ref;
    assert(!table_->lookupTable_.count(ref));
    table_->lookupTable_[ref] = curEntry_;
    stack_.push_back(curEntry_);
    if (prev) {
      prev->children.push_back(curEntry_);
    }
  }

  void pop() {
    assert(!stack_.empty());
    stack_.pop_back();
    if (stack_.empty()) {
      curEntry_ = nullptr;
    } else {
      curEntry_ = stack_.back();
    }
  }

  SymbolTableEntry* cur() const {
    return curEntry_;
  }

  std::unique_ptr<SymbolTable> table_;
  std::vector<SymbolTableEntry*> stack_;
  SymbolTableEntry* curEntry_ = nullptr;

  // Used to determine whether we should add symbols in the context of an
  // assignment (e.g. as a STORE and not a LOAD).
  // TODO AST should populate this information
  bool isAssignmentContext_ = false;
  class AssignmentContextGuard {
   public:
    AssignmentContextGuard(SymbolTableBuilder* builder) : builder_(builder) {
      assert(builder->isAssignmentContext_ == false);
      builder->isAssignmentContext_ = true;
    }
    ~AssignmentContextGuard() noexcept {
      builder_->isAssignmentContext_ = false;
    }

   private:
    SymbolTableBuilder* builder_;
  };
};

void SymbolTableEntry::dump(size_t indent) const {
  std::string kind_;
  switch (kind) {
    case BlockKind::Class:
      kind_ = "class";
      break;
    case BlockKind::Module:
      kind_ = "module";
      break;
    case BlockKind::Function:
      kind_ = "function";
      break;
  }
  fmt::print("Symbol table for '{}' ({})\n", name, kind_);
  for (const auto& pr : symbols) {
    fmt::print("Name: {}\n", pr.first);
    SymbolInfo info = pr.second;
    if (info & SymbolFlag::USE) {
      fmt::print("\tUSE\n");
    }
    if (info & SymbolFlag::DEF_PARAM) {
      fmt::print("\tDEF_PARAM\n");
    }
    if (info & SymbolFlag::DEF_LOCAL) {
      fmt::print("\tDEF_LOCAL\n");
    }
  }
  for (const auto& child : children) {
    child->dump();
  }
}
void SymbolTable::dump() const {
  if (entries_.empty()) {
    fmt::print("Empty symtable\n");
  }
  auto root = entries_.begin()->get();
  fmt::print("{:*^30}\n", "ROOT");
  root->dump();
}

std::unique_ptr<SymbolTable> SymbolTable::build(Mod module) {
  SymbolTableBuilder builder;
  auto st = builder.build(module);
  // st->dump();
  return st;
}

std::unique_ptr<SymbolTable> SymbolTable::build(Def def) {
  SymbolTableBuilder builder;
  auto st = builder.build(def);
  // st->dump();
  return st;
}

SymbolTableEntry* SymbolTable::lookup(TreeRef astNode) {
  return lookupTable_.at(astNode);
}
} // namespace dynamic
} // namespace jit
} // namespace torch
