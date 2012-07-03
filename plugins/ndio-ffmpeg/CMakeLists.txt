cmake_minimum_required(VERSION 2.8)
project(ndio-ffmpeg-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(ROOT_3RDPARTY_DIR ${PROJECT_SOURCE_DIR})

find_package(FFMPEG)
if(FFMPEG_FOUND)
  include_directories(${FFMPEG_INCLUDE_DIRS})

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

  add_definitions(-fPIC)
  add_library(ndio-ffmpeg MODULE ${SRCS} ${HDRS})
  target_link_libraries(ndio-ffmpeg ${FFMPEG_LIBRARIES} libnd)

  ## Copy FFMPEG shared libs to plugin build dir so build functions in place
  #  (Not super sure what to do for non MSVC)
  if(MSVC)
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
      file(COPY ${FFMPEG_SHARED_LIBS} DESTINATION "${PROJECT_BINARY_DIR}/${cfg}") 
    endforeach(cfg)
  else()
    file(COPY ${FFMPEG_SHARED_LIBS} DESTINATION ${PROJECT_BINARY_DIR}) 
  endif()

endif(FFMPEG_FOUND)
