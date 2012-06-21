cmake_minimum_required(VERSION 2.8)
project(ndio-tiff-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
find_package(Mylib REQUIRED)
include_directories(${MYLIB_INCLUDE_DIR})

get_directory_property(ND_SOURCE_DIR PARENT_DIRECTORY)

include_directories(${ND_SOURCE_DIR}) # from parent scope

################################################################################
# SOURCE
################################################################################

file(GLOB SRCS src/*.c)
file(GLOB HDRS src/*.h)

################################################################################
# TARGETS
################################################################################

add_library(ndio-tiff MODULE ${SRCS} ${HDRS})
target_link_libraries(ndio-tiff ${MYLIB_LIBRARIES} libnd)

