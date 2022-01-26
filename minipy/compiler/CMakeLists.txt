add_library(compiler
    Compiler.cpp
    SymbolTable.cpp
    Serialization.cpp
)

target_link_libraries(compiler PUBLIC interpreter)
target_link_libraries(compiler PRIVATE fmt::fmt)

add_subdirectory(test)
