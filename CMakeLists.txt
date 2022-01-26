cmake_minimum_required(VERSION 3.14)

project(minipy C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set colored ninja output
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options (-fcolor-diagnostics)
endif ()

enable_testing()


## Dependencies
# fmt
find_package(fmt)
# gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${PROJECT_SOURCE_DIR})

# hard code asan for now:
# add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
# add_link_options(-fsanitize=address)

add_subdirectory(minipy)
add_subdirectory(examples)
