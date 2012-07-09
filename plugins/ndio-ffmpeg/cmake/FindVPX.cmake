# Locate libvpx libraries
#
# Requires yasm.
# Caller should define an ExternalProject target named yasm, and define
# YASM_ROOT_DIR as the install directory.
#
# This module defines
# VPX_LIBRARY, the name of the library to link against
# VPX_FOUND, if false, do not try to link
# VPX_INCLUDE_DIR, where to find header
#
find_path(VPX_INCLUDE_DIR vpx/vp8.h)
find_library(VPX_LIBRARY NAME vpx)
if(NOT VPX_LIBRARY OR NOT EXISTS ${VPX_LIBRARY})
  include(ExternalProject)
  ExternalProject_Add(libvpx
    DEPENDS yasm
    GIT_REPOSITORY http://git.chromium.org/webm/libvpx.git
    CONFIGURE_COMMAND
        AS=${YASM_ROOT_DIR}/bin/yasm
        <SOURCE_DIR>/configure
          --prefix=<INSTALL_DIR>
          --enable-static
          --enable-pic
          --disable-examples
          --disable-unit-tests
  )
  get_target_property(VPX_ROOT_DIR libvpx _EP_INSTALL_DIR)
  set(VPX_LIBRARY ${VPX_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vpx${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "Path to library." FORCE)
  set(VPX_INCLUDE_DIR ${VPX_ROOT_DIR}/include CACHE PATH "Path to vpx/vp8.h" FORCE)
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VPX DEFAULT_MSG
  VPX_LIBRARY
  VPX_INCLUDE_DIR
)
