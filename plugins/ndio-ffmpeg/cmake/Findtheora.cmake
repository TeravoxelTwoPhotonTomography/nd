# Locate libtheora libraries
# This module defines
# THEORA_LIBRARY, the name of the library to link against
# THEORA_FOUND, if false, do not try to link
# THEORA_INCLUDE_DIR, where to find header
# THEORA_INCLUDE_DIRS, paths to all required headers
#

find_path(THEORA_INCLUDE_DIR theora/theora.h)
find_library(THEORA_LIBRARY     NAME theora)
find_library(THEORA_LIBRARY_ENC NAME theoraenc)
find_library(THEORA_LIBRARY_DEC NAME theoradec)
if(NOT THEORA_LIBRARY OR NOT EXISTS ${THEORA_LIBRARY}) # assume everything else was found ok based on this guy
  ## Try to download and build
  find_package(Ogg) #requires libogg
  if(NOT EXISTS OGG_LIBRARY) #then it needs to be built, need to wait on target
    set(_theora_depends libogg)
  endif()
  include(ExternalProject)
  ExternalProject_Add(libtheora
    DEPENDS ${_theora_depends}
    URL     http://downloads.xiph.org/releases/theora/libtheora-1.1.1.tar.bz2
    URL_MD5 292ab65cedd5021d6b7ddd117e07cd8e
    CONFIGURE_COMMAND <SOURCE_DIR>/configure 
        --prefix=<INSTALL_DIR>
        --with-pic
        --with-ogg=${OGG_ROOT_DIR}
        --disable-oggtest
  )
  get_target_property(THEORA_ROOT_DIR libtheora _EP_INSTALL_DIR)
  set(THEORA_INCLUDE_DIR ${THEORA_ROOT_DIR}/include CACHE PATH "Path to theora/theora.h" FORCE)
  set(THEORA_LIBRARY     ${THEORA_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}theora${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "Path to library." FORCE)
  set(THEORA_LIBRARY_ENC ${THEORA_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}theoraenc${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "Path to library." FORCE)
  set(THEORA_LIBRARY_DEC ${THEORA_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}theoradec${CMAKE_STATIC_LIBRARY_SUFFIX} CACHE FILEPATH "Path to library." FORCE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(THEORA DEFAULT_MSG
  THEORA_INCLUDE_DIR
  THEORA_LIBRARY
  THEORA_LIBRARY_ENC
  THEORA_LIBRARY_DEC
  OGG_LIBRARY
)

if(THEORA_FOUND)
  set(THEORA_LIBRARIES ${THEORA_LIBRARY} ${THEORA_LIBRARY_ENC} ${THEORA_LIBRARY_DEC} ${OGG_LIBRARY})
  set(THEORA_INCLUDE_DIRS ${THEORA_INCLUDE_DIR} ${OGG_INCLUDE_DIR})
endif()
