# SZIP!
#
# Doesn't "find" SZIP so much as it download's and builds it.
# 
include(ExternalProject)
include(FindPackageHandleStandardArgs)

if(NOT TARGET libszip)
  ExternalProject_Add(libszip
    URL        http://www.hdfgroup.org/ftp/lib-external/szip/2.1/src/szip-2.1.tar.gz
    URL_MD5    902f831bcefb69c6b635374424acbead
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    )
endif()

get_target_property(_SZIP_INCLUDE_DIR libszip _EP_SOURCE_DIR)
get_target_property(SZIP_ROOT_DIR     libszip _EP_INSTALL_DIR)

set(SZIP_INCLUDE_DIR ${SZIP_ROOT_DIR}/src CACHE PATH "Path to szlib.h" FORCE)

add_library(szip STATIC IMPORTED)
set_target_properties(szip PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "C"
  IMPORTED_LOCATION         "${SZIP_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}szip${CMAKE_STATIC_LIBRARY_SUFFIX}"
  IMPORTED_LOCATION_RELEASE "${SZIP_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}szip${CMAKE_STATIC_LIBRARY_SUFFIX}"
  IMPORTED_LOCATION_DEBUG   "${SZIP_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}libszip_debug${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
add_dependencies(szip libszip)

set(SZIP_LIBRARY szip)

find_package_handle_standard_args(SZIP DEFAULT_MSG SZIP_INCLUDE_DIR SZIP_LIBRARY)
