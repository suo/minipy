add_library(minipy
    compiler.cpp
    instruction.cpp
    interpreter.cpp
    symbol_table.cpp
    types.cpp
    Obj.cpp
)

add_subdirectory(common)
add_subdirectory(jitparse)
target_link_libraries(minipy parser fmt::fmt)

