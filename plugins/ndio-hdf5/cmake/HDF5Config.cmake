#
# HDF5
# Doesn't "find" HDF5 so much as it download's and builds it.
# 
# TODO
#  - add zlib and szip support
#
include(ExternalProject)
include(FindPackageHandleStandardArgs)

## Override the HDF5 "Use external" cmake scripts.
#  Use our own configs.  Since we're doing static linking, all the libraries can
#  get linked together at the end.
#
#  I could never get their "Use external" scripts to work quite right...
#
#  Two downsides to the approach here.
#  1. The HDF5 command line tools won't build.
#  2. SZIP and ZLIB get downloaded and built twice.  Once by this package, and
#     once by HDF5.
#  There are probably ways to avoid these.
#
if(NOT TARGET libhdf5)
  ## Patch to deal with broken MSVC static build: see ZLIBConfig.cmake for more details.
  ## Do the patch config here so it only happens once.
  find_package(SZIP CONFIG PATHS cmake CONFIGS SZIPConfig.cmake)
  find_package(ZLIB CONFIG PATHS cmake CONFIGS ZLIBConfig.cmake)

  ExternalProject_Add(libhdf5
    URL        http://www.hdfgroup.uiuc.edu/ftp/pub/outgoing/hdf5/snapshots/v19/hdf5-1.9.125.tar.gz
    URL_MD5    60246a2323058dd3ad832dcb2ca4fcda
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               #-DCMAKE_PREFIX_PATH:PATH=${PROJECT_SOURCE_DIR}/cmake ## Use our custom ZLIB and SZIB configs
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  #             -DHDF5_BUILD_TOOLS:BOOL=TRUE ## The way we do the szip and zlib dependencies excludes building the tools
               -DHDF5_ENABLE_SZIP_ENCODING:BOOL=1
               -DHDF5_ENABLE_SZIP_SUPPORT:BOOL=1
               -DSZIP_USE_EXTERNAL:BOOL=1
               -DSZIP_FOUND:BOOL=${SZIP_FOUND}
               -DSZIP_INCLUDE_DIR:STRING=${SZIP_INCLUDE_DIR}
               -DSZIP_LIBRARIES:STRING=${SZIP_LIBRARIES}
               -DHDF5_ENABLE_Z_LIB_SUPPORT:BOOL=1
               -DHDF5_Z_LIB_USE_EXTERNAL:BOOL=1
               -DH5_HAVE_ZLIB_H:BOOL=TRUE               
               -DH5_ZLIB_HEADER:STRING=zlib.h
               -DZLIB_INCLUDE_DIRS:STRING=${ZLIB_INCLUDE_DIRS}
               -DZLIB_LIBRARIES:STRING=${ZLIB_LIBRARIES}
               -DBUILD_STATIC_PIC:BOOL=TRUE
               -DCMAKE_C_FLAGS:STRING=-fPIC
    )
  add_dependencies(libhdf5 libszip libz)
endif()

get_target_property(_HDF5_INCLUDE_DIR libhdf5 _EP_SOURCE_DIR)
get_target_property(HDF5_ROOT_DIR     libhdf5 _EP_INSTALL_DIR)

set(HDF5_INCLUDE_DIR ${HDF5_ROOT_DIR}/include CACHE PATH "Path to hdf5.h" FORCE)

add_library(hdf5 STATIC IMPORTED)

if(WIN32)
  set(LIBRARY_PREFIX lib)
  set(DEBUG_LIBRARY_SUFFIX _D)
else()
  set(LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(DEBUG_LIBRARY_SUFFIX _debug)
endif()

set_target_properties(hdf5 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "C"
  IMPORTED_LOCATION         "${HDF5_ROOT_DIR}/lib/${LIBRARY_PREFIX}hdf5${CMAKE_STATIC_LIBRARY_SUFFIX}"
  IMPORTED_LOCATION_RELEASE "${HDF5_ROOT_DIR}/lib/${LIBRARY_PREFIX}hdf5${CMAKE_STATIC_LIBRARY_SUFFIX}"
  IMPORTED_LOCATION_DEBUG   "${HDF5_ROOT_DIR}/lib/${LIBRARY_PREFIX}hdf5${DEBUG_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
add_dependencies(hdf5 libhdf5 z szip)

set(HDF5_LIBRARY   hdf5 z szip)
set(HDF5_LIBRARIES hdf5 z szip)            # set plural forms bc I can't remember 
set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR}) # which ones to use

find_package_handle_standard_args(HDF5 DEFAULT_MSG
  HDF5_INCLUDE_DIR
  HDF5_LIBRARY
)
