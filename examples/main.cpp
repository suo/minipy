#include <fmt/format.h>
#include "minipy/compiler/compiler.h"
#include "minipy/interpreter/interpreter.h"
#include "minipy/jitparse/parser.h"

static constexpr auto src = R"SCRIPT(
def foo(x):
    if x > 5:
        print("greater than 5")
    else:
        print("lte 5")

foo(6)
foo(4)
)SCRIPT";

using namespace ::minipy;

int main() {
  fmt::print("Compiling code:\n{}\n", src);
  Parser p(std::make_shared<Source>(src));
  auto mod = p.parseModule();

  auto code = emit(mod);
  Obj print = c10::make_intrusive<BuiltinFunction>("print", [](Obj args) {
    fmt::print("{}\n", args.str().toStringRef());
    return Obj();
  });

  FrameObject frame;
  frame.globals.emplace("print", print);

  auto interpreter = Interpreter(*code, frame);
  interpreter.run();

  return 0;
}
