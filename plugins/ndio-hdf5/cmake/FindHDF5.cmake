#
# HDF5
# Doesn't "find" HDF5 so much as it download's and builds it.
# 
# TODO
#  - add zlib and szip support
#
include(ExternalProject)
include(FindPackageHandleStandardArgs)

ExternalProject_Add(libhdf5
  URL        http://www.hdfgroup.uiuc.edu/ftp/pub/outgoing/hdf5/snapshots/v19/hdf5-1.9.125.tar.gz
  URL_MD5    60246a2323058dd3ad832dcb2ca4fcda
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DHDF5_BUILD_TOOLS:BOOL=FALSE
  )

get_target_property(HDF5_ROOT_DIR     libhdf5 _EP_INSTALL_DIR)
set(HDF5_INCLUDE_DIR ${HDF5_ROOT_DIR}/include)

#set(HDF5_INCLUDE_DIR ${HDF5_ROOT_DIR}/include CACHE PATH "Path to hdf5.h" FORCE)

add_library(hdf5 STATIC IMPORTED)
set_target_properties(hdf5 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "C"
  IMPORTED_LOCATION "${HDF5_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hdf5_debug${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
add_dependencies(hdf5 libhdf5)

set(HDF5_LIBRARY hdf5)

#define the plural forms bc I can never remember which to use
set(HDF5_LIBRARIES hdf5)
set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})

find_package_handle_standard_args(HDF5 DEFAULT_MSG
  HDF5_LIBRARY
  HDF5_INCLUDE_DIR
)