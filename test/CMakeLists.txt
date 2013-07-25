#
# Testing
#
find_package(GTEST PATHS cmake)
find_package(ND COMPONENTS ndio-hdf5) # for debugging mostly
find_package(Threads)
if(GTEST_FOUND)
  enable_testing()
  include_directories(${PROJECT_SOURCE_DIR})
  include_directories(${GTEST_INCLUDE_DIR})
  file(GLOB TEST_SOURCES *.cc)
  add_executable(test-libnd ${TEST_SOURCES})
  target_link_libraries(test-libnd
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    nd
    )
  add_dependencies(test-libnd ndio-hdf5)
  add_test(TestLibND test-libnd)
  add_definitions(-DGTEST_LINKED_AS_SHARED_LIBRARY=1)
  gtest_copy_shared_libraries(test-libnd)
  nd_copy_plugins_to_target(test-libnd ndio-hdf5)
  install(TARGETS test-libnd EXPORT libnd-targets RUNTIME DESTINATION bin/test)
endif()
