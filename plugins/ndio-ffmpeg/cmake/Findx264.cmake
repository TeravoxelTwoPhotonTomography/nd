# Locate libx264 library  
# This module defines
# X264_LIBRARY, the name of the library to link against
# X264_FOUND, if false, do not try to link
# X264_INCLUDE_DIR, where to find header
#

find_path(X264_INCLUDE_DIR x264.h)
find_library(X264_LIBRARY NAMES x264)
if(NOT X264_LIBRARY OR NOT EXISTS ${X264_LIBRARY})
  include(ExternalProject)
  ExternalProject_Add(libx264
    URL ftp://ftp.videolan.org/pub/x264/snapshots/last_x264.tar.bz2
    #GIT_REPOSITORY https://git.videolan.org/x264.git
    CONFIGURE_COMMAND <SOURCE_DIR>/configure
          --prefix=<INSTALL_DIR>
          --enable-static
          --enable-shared
  )
  get_target_property(X264_ROOT_DIR libx264 _EP_INSTALL_DIR)
  set(X264_INCLUDE_DIR ${X264_ROOT_DIR}/include CACHE PATH "Path to x264.h" FORCE)
  set(X264_LIBRARY     ${X264_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}x264${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "Path to library." FORCE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(X264 DEFAULT_MSG
  X264_LIBRARY
  X264_INCLUDE_DIR
)
