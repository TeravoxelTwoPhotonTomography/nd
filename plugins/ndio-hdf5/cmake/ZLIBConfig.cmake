# ZLIB!
#
# Doesn't "find" ZLIB so much as it download's and builds it.
# 
include(ExternalProject)
include(FindPackageHandleStandardArgs)

if(NOT TARGET libz)
  ExternalProject_Add(libz
    URL        http://zlib.net/zlib-1.2.7.tar.gz
    URL_MD5    60df6a37c56e7c1366cca812414f7b85
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
    BUILD_IN_SOURCE   TRUE
    )
endif()

get_target_property(_ZLIB_INCLUDE_DIR libz _EP_SOURCE_DIR)
get_target_property(ZLIB_ROOT_DIR     libz _EP_INSTALL_DIR)

set(ZLIB_INCLUDE_DIR ${ZLIB_ROOT_DIR} CACHE PATH "Path to zlib.h" FORCE)

add_library(z STATIC IMPORTED)
set_target_properties(z PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "C"
  IMPORTED_LOCATION "${ZLIB_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
add_dependencies(z libz)
set(ZLIB_LIBRARY z)
find_package_handle_standard_args(ZLIB DEFAULT_MSG ZLIB_INCLUDE_DIR ZLIB_LIBRARY)
