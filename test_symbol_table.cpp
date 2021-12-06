#include <gtest/gtest.h>

#include <minipy/symbol_table.h>
#include <minipy/jitparse/parser.h>

namespace torch {
namespace jit {
namespace dynamic {

TEST(SymbolTable, DuplicateParams) {
  static constexpr auto moduleSource = R"SCRIPT(
def foo(x, x, x):
    return x

foo(9, 9, 9)
)SCRIPT";
  Parser p(std::make_shared<Source>(moduleSource));
  auto moduleAst = p.parseModule();
  EXPECT_THROW(SymbolTable::build(moduleAst), std::exception);
}

TEST(SymbolTable, TestBasic) {
  static constexpr auto moduleSource = R"SCRIPT(
def foo(x):
    y = torch.add(x, x) + x
    x = y + y
    return x

foo(9)
)SCRIPT";
  Parser p(std::make_shared<Source>(moduleSource));
  const auto moduleAst = p.parseModule();
  auto st = SymbolTable::build(moduleAst);

  // In the module scope, there should be `foo` symbol defined locally
  SymbolTableEntry* ste = st->lookup(moduleAst.tree());
  EXPECT_TRUE(ste->symbols["foo"] & SymbolFlag::DEF_LOCAL);
  EXPECT_TRUE(ste->symbols["foo"] & SymbolFlag::USE);

  ste = st->lookup(moduleAst.body()[0].tree());
  EXPECT_TRUE(ste->symbols.count("x"));
  EXPECT_TRUE(ste->symbols["x"] & SymbolFlag::DEF_PARAM);
  EXPECT_TRUE(ste->symbols["x"] & SymbolFlag::USE);

  EXPECT_TRUE(ste->symbols.count("torch"));
  EXPECT_TRUE(ste->symbols["torch"] & SymbolFlag::USE);

  EXPECT_TRUE(ste->symbols.count("y"));
  EXPECT_TRUE(ste->symbols["y"] & SymbolFlag::USE);
  EXPECT_TRUE(ste->symbols["y"] & SymbolFlag::DEF_LOCAL);
}

// SymbolTable should work on function defs only
TEST(SymbolTable, TestDef) {
  static constexpr auto defSource = R"SCRIPT(
def foo(x):
    y = torch.add(x, x) + x
    x = y + y
    return x
)SCRIPT";
  Parser p(std::make_shared<Source>(defSource));
  const auto defAst = Def(p.parseFunction(/*is_method=*/false));
  const auto st = SymbolTable::build(defAst);

  SymbolTableEntry* ste = st->lookup(defAst.tree());
  EXPECT_TRUE(ste->symbols.count("x"));
  EXPECT_TRUE(ste->symbols["x"] & SymbolFlag::DEF_PARAM);
  EXPECT_TRUE(ste->symbols["x"] & SymbolFlag::USE);

  EXPECT_TRUE(ste->symbols.count("torch"));
  EXPECT_TRUE(ste->symbols["torch"] & SymbolFlag::USE);

  EXPECT_TRUE(ste->symbols.count("y"));
  EXPECT_TRUE(ste->symbols["y"] & SymbolFlag::USE);
  EXPECT_TRUE(ste->symbols["y"] & SymbolFlag::DEF_LOCAL);
}
} // namespace dynamic
} // namespace jit
} // namespace torch
