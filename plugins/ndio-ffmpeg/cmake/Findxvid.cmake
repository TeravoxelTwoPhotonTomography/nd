# Locate xvid library  
# This module defines
# XVID_LIBRARY, the name of the library to link against
# XVID_FOUND, if false, do not try to link
# XVID_INCLUDE_DIR, where to find header
#

find_path(XVID_INCLUDE_DIR xvid.h)
find_library(XVID_LIBRARY NAMES xvidcore)

if(NOT XVID_LIBRARY OR NOT EXISTS ${XVID_LIBRARY})
  include(ExternalProject)
  ExternalProject_Add(libxvid
      URL               http://downloads.xvid.org/downloads/xvidcore-1.3.2.tar.gz
      CONFIGURE_COMMAND <SOURCE_DIR>/build/generic/configure --prefix=<INSTALL_DIR>
      BUILD_COMMAND     make -C <SOURCE_DIR>/build/generic
      INSTALL_COMMAND   make -C <SOURCE_DIR>/build/generic install
      BUILD_IN_SOURCE   TRUE
  )
  get_target_property(XVID_ROOT_DIR libxvid _EP_INSTALL_DIR)
  set(XVID_INCLUDE_DIR ${XVID_ROOT_DIR}/include CACHE PATH "Location of xvid.h" FORCE)
  set(XVID_LIBRARY     ${XVID_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}xvidcore${CMAKE_STATIC_LIBRARY_SUFFIX} 
                       CACHE FILEPATH "Location of library." FORCE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XVID DEFAULT_MSG
  XVID_INCLUDE_DIR
  XVID_LIBRARY
)
