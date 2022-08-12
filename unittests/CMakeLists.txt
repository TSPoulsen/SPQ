enable_testing()

include(${SPQ_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND SPQ_UNITTESTS 
  math_utils
  loss
  kmeans
  product_quantizer
)

set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -Wall -Wextra -O3")

foreach(unittest IN LISTS SPQ_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME})
  target_include_directories(${unittest} PRIVATE ${SPQ_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()

foreach(unittest IN LISTS SPQ_UNITTESTS)
  set(unittest_avx  ${unittest}_avx)
  add_executable(${unittest_avx} ${unittest}.cpp)
  set_target_properties(${unittest_avx} PROPERTIES COMPILE_FLAGS "${DEFAULT_CXX_FLAGS} -march=native")
  target_link_libraries(${unittest_avx} ${PROJECT_NAME})
  target_include_directories(${unittest_avx} PRIVATE ${SPQ_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest_avx})
endforeach()
