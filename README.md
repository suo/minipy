# minipy

prototype minimal python runtime

To build:

```sh
# fmt is a dependency
conda install -c conda-forge fmt
# ninja is a dev dependency
conda install ninja

# the way I am building, from project root
CC=clang CXX=clang++ cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -S . -B build
cmake --build build

# run the example
build/examples/example

# run tests
cd build && ctest
```

Builds use ASAN by default.
