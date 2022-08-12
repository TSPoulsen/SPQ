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

# install header
#message(${CMAKE_CURRENT_SOURCE_DIR})
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/spq)

# export target
#set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME ${PROJECT_NAME})

#export(
  #TARGETS ${PROJECT_NAME} 
  #NAMESPACE ${PROJECT_NAME}:: 
  #FILE ${PROJECT_NAME}Targets.cmake
#)
#export(PACKAGE ${PROJECT_NAME})
#
#install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
#install(
  #EXPORT ${PROJECT_NAME}Targets 
  #NAMESPACE ${PROJECT_NAME}:: 
  #DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})


#set(DEFAULT_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O3")
#set_target_properties(test_no_avx PROPERTIES COMPILE_FLAGS "${DEFAULT_CXX_FLAGS}")
#set_target_properties(test_avx PROPERTIES COMPILE_FLAGS "${DEFAULT_CXX_FLAGS} -march=native")

#include_directories("include")
#include_directories("tests")

#add_custom_target(test
    #COMMAND test_avx
    #COMMAND test_no_avx
    #DEPENDS test_avx
    #DEPENDS test_no_avx
    #WORKING_DIRECTORY ${CMAKE_PROJECT_DIR})