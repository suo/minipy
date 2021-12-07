add_library(minipy INTERFACE)

add_subdirectory(common)
add_subdirectory(jitparse)
add_subdirectory(interpreter)
add_subdirectory(compiler)
target_link_libraries(minipy INTERFACE parser interpreter compiler)
