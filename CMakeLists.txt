cmake_minimum_required(VERSION 3.5)

project(minipy C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

find_package(fmt)
include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(minipy)
add_executable(test main.cpp)
target_link_libraries(test minipy)
