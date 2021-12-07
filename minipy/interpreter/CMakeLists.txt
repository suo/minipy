add_library(interpreter
    instruction.cpp
    interpreter.cpp
    types.cpp
    Obj.cpp
)

target_link_libraries(interpreter PRIVATE fmt::fmt)
