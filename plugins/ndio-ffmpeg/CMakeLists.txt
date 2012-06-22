cmake_minimum_required(VERSION 2.8)
project(ndio-ffmpeg-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(ffmpeg)
if(FFMPEG_FOUND)
  include_directories(${FFMPEG_INCLUDE_DIR})

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

  add_library(ndio-ffmpeg MODULE ${SRCS} ${HDRS})
  target_link_libraries(ndio-ffmpeg ${FFMPEG_LIBRARIES} libnd)


endif(FFMPEG_FOUND)
