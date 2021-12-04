#pragma once
#include <minipy/jitparse/tree.h>
#include <minipy/jitparse/tree_views.h>
#include <memory>

namespace torch {
namespace jit {

struct Decl;
struct ParserImpl;
struct Lexer;

Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method);

struct Parser {
  explicit Parser(const std::shared_ptr<SourceView>& src);
  TreeRef parseFunction(bool is_method);
  TreeRef parseClass();
  Mod parseModule();
  Decl parseTypeComment();
  Expr parseExp();
  Lexer& lexer();
  ~Parser();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

} // namespace jit
} // namespace torch
