enable_testing()

include(${SPQ_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND SPQ_UNITTESTS 
  math_utils
  #loss
  #kmeans
  #product_quantizer
)

foreach(unittest IN LISTS SPQ_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME})
  target_include_directories(${unittest} PRIVATE ${SPQ_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()
