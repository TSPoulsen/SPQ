cmake_minimum_required(VERSION 3.12)
project(SimplyPQ CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(SPQ_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

include(CTest)

include_directories(${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)

add_subdirectory(unittests)