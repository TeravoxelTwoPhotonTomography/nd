cmake_minimum_required(VERSION 2.8)
project(nd)

set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACAKGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0a)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "N-Dimensional Scalar Volume Library"
  )

macro(show v)
  message("${v} is ${${v}}")
endmacro()

set(CMAKE_INSTALL_RPATH $ORIGIN) ## Probably need to change this to @rpath or @loaderpath on os x
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}) ## OSX specific(untested)
#show(CMAKE_INSTALL_PREFIX)
#show(CMAKE_INSTALL_RPATH)
#show(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
#show(CMAKE_BUILD_WITH_INSTALL_RPATH)
#show(CMAKE_INSTALL_NAME_DIR)
#show(CMAKE_SKIP_RPATH)
#show(CMAKE_SKIP_BUILD_RPATH)
#show(CMAKE_SKIP_INSTALL_RPATH)

################################################################################
# CONFIG
################################################################################

### GTest (as an ExternalProject)
#   do this before anything else so gtest gets built in the root dir
find_package(GTEST CONFIG PATHS cmake)

### NDIO plugins
set(NDIO_PLUGIN_PATH  plugins)
set(ND_TEST_DATA_PATH ${PROJECT_SOURCE_DIR}/test/data)

### Windows shell lightweight utility functions - for plugin search
if(WIN32)
find_library(SHLWAPI Shlwapi.lib) 
else()
set(SHLWAPI)
endif()

### CUDA
set(HAVE_CUDA 0)
find_package(CUDA 4.0)
if(CUDA_FOUND)
  set(HAVE_CUDA 1)
  include(cmake/CheckCudaComputeCaps.cmake)
  show(CUDA_GENERATE_CODE)
  include_directories(${CUDA_INCLUDE_DIRS})
  cuda_build_clean_target()
  set(cuda_build_cubin on)
  set(CUOPTS
    -Xcompiler -fPIC --generate-code ${CUDA_GENERATE_CODE}
    DEBUG -g -G
    RELEASE --use_fast_math
    RELWITHDEBINFO --use_fast_math;-g
    MINSIZEREL --use_fast_math
    )
endif()

################################################################################
# SOURCE
################################################################################

file(GLOB SRCS  src/*.c src/*.cc src/io/*.c src/affine/*.c src/conv/*.c)
file(GLOB HDRS  *.h src/*.h src/io/*. src/conv/*.h)
file(GLOB CUSRC src/affine/*.cu src/conv/*.cu src/generic/*.cu)
include_directories(${PROJECT_SOURCE_DIR})


# PUBLIC HEADERS
install(
  FILES   nd.h
  #EXPORT  libnd-targets 
  DESTINATION include/nd)
install(
  FILES 
    src/ops.h
    src/core.h
    src/io.h
  #EXPORT libnd-targets
  DESTINATION include/nd/src)

if(CUDA_FOUND)
  #add_executable(cuda_compute_capability cmake/cuda_compute_capability.c)
  #target_link_libraries(cuda_compute_capability ${CUDA_LIBRARIES})
  cuda_compile(CUFILES STATIC ${CUSRC} OPTIONS ${CUOPTS})
endif()

################################################################################
# config.h
################################################################################
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
include_directories(${PROJECT_BINARY_DIR})

################################################################################
# TARGETS
################################################################################

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
  ${CUDA_CUFFT_LIBRARIES}
  ${CMAKE_DL_LIBS}
  ${SHLWAPI}
)
set_target_properties(nd PROPERTIES 
  POSITION_INDEPENDENT_CODE TRUE
  PUBLIC_HEADERS            "${HDRS}"
)
install(TARGETS nd EXPORT libnd-targets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

### APPS
# add_subdirectory(app/ndconvert) # FIXME

###############################################################################
#  Testing
###############################################################################
add_subdirectory(test)

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
install(EXPORT libnd-targets DESTINATION cmake)
export(TARGETS nd FILE nd-config-targets.cmake )
export(PACKAGE nd)
configure_file(nd-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/nd-config.cmake @ONLY)
