# TODO: copy test data to build directory
# TODO: copy plugins to proper place in build
#
cmake_minimum_required(VERSION 2.8)
project(nd)

set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACAKGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0a)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "N-Dimensional Scalar Volume Library"
  )

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(ROOT_3RDPARTY_PATH
  $ENV{USERPROFILE}/Desktop/src/3rdParty
  ~/src/3rdParty
  )

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################

set(NDIO_PLUGIN_PATH  plugins)
set(ND_TEST_DATA_PATH ${PROJECT_SOURCE_DIR}/test/data)
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
include_directories(${PROJECT_BINARY_DIR})

################################################################################
# SOURCE
################################################################################

file(GLOB SRCS src/*.c src/io/*.c)
file(GLOB HDRS src/*.h src/io/*.h)

################################################################################
# TARGETS
################################################################################

add_library(libnd
  ${SRCS}
  ${HDRS}
  config.h.in
  ${PROJECT_BINARY_DIR}/config.h
  )

# ndio plugins
file(GLOB PLUGINS plugins/*)
foreach(PLUGIN ${PLUGINS})
  add_subdirectory(${PLUGIN})
endforeach(PLUGIN)

###############################################################################
#  Testing
###############################################################################

find_path(GTEST_ROOT
  CMakeLists.txt
  HINTS ${ROOT_3RDPARTY_PATH}
  PATH_SUFFIXES
    gtest-1.6.0
  DOC "The root directory of the gtest install prefix")
find_package(GTest)
find_package(Threads)
if(GTEST_FOUND)
  enable_testing()
  include_directories(${PROJECT_SOURCE_DIR})
  include_directories(${GTEST_INCLUDE_DIRS})
  file(GLOB TEST_SOURCES test/*.cc)
  add_executable(libnd-tests
    ${TEST_SOURCES}
    )
  target_link_libraries(libnd-tests
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    libnd
    )
  add_dependencies(libnd-tests ndio-tiff)
  add_test(TestLibND libnd-tests)
endif()

###############################################################################
#  Documentation
###############################################################################

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
                 @ONLY
                 )
  add_custom_target(libnd-docs
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT           "Generating API Docs"
    VERBATIM
    )
endif()

