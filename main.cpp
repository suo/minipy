#include <fmt/format.h>
#include <minipy/compiler.h>
#include <minipy/jitparse/parser.h>

static constexpr auto src = R"SCRIPT(
def foo(x):
    if x > 5:
        print("greater than 5")
    else:
        print("lte 5")

foo(6)
foo(4)
)SCRIPT";

using namespace ::torch::jit::dynamic;
using namespace ::torch::jit;

int main() {
  fmt::print("Compiling code:\n{}\n", src);
  Parser p(std::make_shared<Source>(src));
  auto mod = p.parseModule();

  emit(mod);

  return 0;
}
