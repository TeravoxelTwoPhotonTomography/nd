cmake_minimum_required(VERSION 2.8)
project(nd)

set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACAKGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0a)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "N-Dimensional Scalar Volume Library"
  )

#set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE)

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################

### NDIO plugins
set(NDIO_PLUGIN_PATH  plugins)
set(ND_TEST_DATA_PATH ${PROJECT_SOURCE_DIR}/test/data)
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
include_directories(${PROJECT_BINARY_DIR})

### CUDA
find_package(CUDA 4.0)
include_directories(${CUDA_INCLUDE_DIRS})
cuda_build_clean_target()
set(cuda_build_cubin on)
set(CUOPTS)

################################################################################
# SOURCE
################################################################################

file(GLOB SRCS  src/*.c src/io/*.c src/affine/*.c)
file(GLOB HDRS  src/*.h src/io/*.h)
file(GLOB CUSRC src/affine/*.cu)

cuda_compile(CUFILES STATIC ${CUSRC} ${CUOPTS})

################################################################################
# TARGETS
################################################################################

add_definitions(-fPIC)

add_library(nd
  ${SRCS}
  ${HDRS}
  ${CUSRC}
  ${CUFILES}
  config.h.in
  ${PROJECT_BINARY_DIR}/config.h
  )
target_link_libraries(nd
  ${CUDA_LIBRARIES}
  ${CMAKE_DL_LIBS}
)

# ndio plugins
file(GLOB PLUGINS plugins/ndio*)
foreach(PLUGIN ${PLUGINS})
  add_subdirectory(${PLUGIN})
endforeach(PLUGIN)

###############################################################################
#  Testing
###############################################################################

include(GenerateGoogleTest)
find_package(Threads)
if(GTEST_FOUND)
  enable_testing()
  include_directories(${PROJECT_SOURCE_DIR})
  include_directories(${GTEST_INCLUDE_DIR})
  file(GLOB TEST_SOURCES test/*.cc)
  add_executable(libnd-tests
    ${TEST_SOURCES}
    )
  target_link_libraries(libnd-tests
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    nd
    )
  add_dependencies(libnd-tests
    gtest
    ndio-tiff
    ndio-ffmpeg)
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
    ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT           "Generating API Docs"
    VERBATIM
    )
endif()

###############################################################################
#  Install
###############################################################################
install(TARGETS nd libnd-tests
  EXPORT libnd-targets
  RUNTIME DESTINATION bin/
  LIBRARY DESTINATION lib/
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include/nd
)
export(TARGETS nd libnd-tests FILE libnd-targets.cmake)
install(EXPORT libnd-targets DESTINATION cmake/)


