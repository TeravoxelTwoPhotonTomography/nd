## Find libogg
#  Will attempt to download and build from source if the library isn't found.
#
find_library(OGG_LIBRARY NAMES ogg)
find_path(OGG_INCLUDE_DIR ogg/ogg.h)
if(NOT OGG_LIBRARY OR NOT EXISTS ${OGG_LIBRARY})
  # Try external project
  include(ExternalProject)
  ExternalProject_Add(libogg
    URL      http://downloads.xiph.org/releases/ogg/libogg-1.3.0.tar.gz
    URL_MD5  0a7eb40b86ac050db3a789ab65fe21c2
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --with-pic
  )
  get_target_property(OGG_ROOT_DIR libogg _EP_INSTALL_DIR)
  set(OGG_INCLUDE_DIR ${OGG_ROOT_DIR}/include CACHE PATH "Path to ogg/ogg.h" FORCE)
  set(OGG_LIBRARY     ${OGG_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ogg${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "Path to library." FORCE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OGG DEFAULT_MSG
  OGG_INCLUDE_DIR
  OGG_LIBRARY
)
