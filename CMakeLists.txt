cmake_minimum_required(VERSION 3.10)
project(JPQ CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_executable(test tests/main.cpp)
add_executable(test_avx tests/main.cpp)

set(DEFAULT_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O3")
set_target_properties(test PROPERTIES COMPILE_FLAGS "${DEFAULT_CXX_FLAGS}")
set_target_properties(test_avx PROPERTIES COMPILE_FLAGS "${DEFAULT_CXX_FLAGS} -march=native")

include_directories("include")
include_directories("tests/include")