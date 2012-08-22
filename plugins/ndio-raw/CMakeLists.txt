cmake_minimum_required(VERSION 2.8)
project(ndio-raw-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################
get_directory_property(ND_SOURCE_DIR PARENT_DIRECTORY)
include_directories(${ND_SOURCE_DIR})

##############################################################################
# SOURCE
##############################################################################

file(GLOB SRCS src/*.c)
file(GLOB HDRS src/*.h)

##############################################################################
# TARGETS
##############################################################################

add_definitions(-fPIC)
add_library(ndio-raw MODULE ${SRCS} ${HDRS})
target_link_libraries(ndio-raw nd)

##############################################################################
#  Testing
##############################################################################

include(GenerateGoogleTest)
file(GLOB TEST_SOURCES test/*.cc)
if(GTEST_FOUND)
  find_package(Threads)
  enable_testing()
  include_directories(${PROJECT_SOURCE_DIR})
  include_directories(${GTEST_INCLUDE_DIR})
  add_executable(ndio-raw-tests ${TEST_SOURCES})
  target_link_libraries(ndio-raw-tests
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    nd
    )
  add_dependencies(ndio-raw-tests gtest)
  add_test(TestNDIOSeries ndio-raw-tests)
  install(TARGETS ndio-raw-tests RUNTIME DESTINATION bin)
endif()

##############################################################################
#  Install
##############################################################################
install(TARGETS ndio-raw
  EXPORT ndio-raw-targets
  RUNTIME DESTINATION bin/${NDIO_PLUGIN_PATH}
  LIBRARY DESTINATION bin/${NDIO_PLUGIN_PATH}
)
export(TARGETS ndio-raw FILE ndio-raw-targets.cmake)
install(EXPORT ndio-raw-targets DESTINATION cmake)

################################################################################
#  Documentation
################################################################################

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
                 @ONLY
                 )
  add_custom_target(ndio-raw-docs
    ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT           "Generating API Docs"
    VERBATIM
    )
endif()


