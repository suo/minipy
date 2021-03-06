#include "minipy/jitparse/lexer.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace minipy {

static const std::unordered_map<int, int> binary_prec = {
    {TK_IF, 1},
    {TK_FOR, 1},
    {TK_AND, 2},
    {TK_OR, 2},
    // reserve a level for unary not
    {TK_IN, 4},
    {TK_NOTIN, 4},
    {'<', 4},
    {'>', 4},
    {TK_IS, 4},
    {TK_ISNOT, 4},
    {TK_EQ, 4},
    {TK_LE, 4},
    {TK_GE, 4},
    {TK_NE, 4},
    {'|', 5},
    {'^', 6},
    {'&', 7},
    {TK_LSHIFT, 8},
    {TK_RSHIFT, 8},
    {'+', 9},
    {'-', 9},
    {'*', 10},
    {'/', 10},
    {TK_FLOOR_DIV, 10},
    {'%', 10},
    {'@', 10},
    {TK_POW, 11},
};

static const std::unordered_map<int, int> unary_prec = {
    {TK_NOT, 3},
    {'~', 3},
    {'-', 10},
    {'*', 10},
};

bool SharedParserData::isUnary(int kind, int* prec) {
  auto it = unary_prec.find(kind);
  if (it != unary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}
bool SharedParserData::isBinary(int kind, int* prec) {
  auto it = binary_prec.find(kind);
  if (it != binary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}

int stringToKind(const std::string& str) {
  static std::once_flag init_flag;
  static std::unordered_map<std::string, int> str_to_kind;
  std::call_once(init_flag, []() {
    for (char tok : std::string(valid_single_char_tokens))
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      str_to_kind[std::string(1, tok)] = tok;
#define DEFINE_CASE(tok, _, str) \
  if (std::string(str) != "")    \
    str_to_kind[str] = tok;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
  });
  try {
    return str_to_kind.at(str);
  } catch (std::out_of_range& err) {
    throw std::out_of_range("unknown token in stringToKind");
  }
}

std::string kindToString(int kind) {
  if (kind < 256)
    return std::string(1, kind);
  switch (kind) {
#define DEFINE_CASE(tok, str, _) \
  case tok:                      \
    return str;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("Unknown kind: " + std::to_string(kind));
  }
}

SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}

} // namespace minipy
