# Include this to generate and build FFMPEG
# INPUT
#   FFMPEG_GIT_REPOSITORY     has default
#   FFMPEG_GIT_BRANCH         has deefalt
#
# OUTPUT
#   FFMPEG_FOUND
#   FFMPEG_LIBRARIES
#   FFMPEG_INCLUDE_DIR

##
## HANDLE INPUT VARIABLES
##
macro(_ffmpeg_setdefault var val)
  if(NOT ${var})
    set(${var} ${val})
  endif()
endmacro(_ffmpeg_setdefault)

_ffmpeg_setdefault(FFMPEG_GIT_REPOSITORY https://github.com/FFmpeg/FFmpeg.git)
_ffmpeg_setdefault(FFMPEG_GIT_BRANCH     n0.11.1)

show(FFMPEG_GIT_REPOSITORY)
show(FFMPEG_GIT_BRANCH)

##
## REQUIRED DEPENDENCIES
##
include(ExternalProject)
ExternalProject_Add(yasm
  URL               http://www.tortall.net/projects/yasm/releases/yasm-1.2.0.tar.gz
  URL_MD5           4cfc0686cf5350dd1305c4d905eb55a6
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
  )
get_target_property(YASM_ROOT_DIR yasm _EP_INSTALL_DIR)

if(APPLE) #FFMPEG may rely on these Frameworks
  find_library(CF_LIBS  CoreFoundation)
  find_library(VDA_LIBS VideoDecodeAcceleration)
  find_library(CV_LIBS  CoreVideo)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${CF_LIBS} ${VDA_LIBS} ${CV_LIBS})
endif()

##
## OPTIONAL DEPENDENCIES
##

include(FeatureSummary)

##
# Modifies FFMPEG_DEP_LIBRARIES, _ffmpeg_conf, _ffmpeg_paths, and _ffmpeg_deps.
# Adds libraries and configure lines if a library is found.
# If <confname> is false, will only add include directories and feature description.
#   The library won't be "enabled" or "disabled" in the FFMPEG config.
macro(_ffmpeg_maybe_add name confname description url)
  find_package(${name})
  set_package_properties(${name} PROPERTIES
    DESCRIPTION ${description}
    URL         ${url}
  )
  string(TOUPPER ${name} uname)
  add_feature_info(${name} ${uname}_FOUND ${description})
  if(${uname}_FOUND)
    include_directories(${uname}_INCLUDE_DIR)
    set(_ffmpeg_conf ${_ffmpeg_conf} --enable-${confname})
    if(${uname}_INCLUDE_DIRS)
      foreach(dir ${${uname}_INCLUDE_DIRS})
        set(_ffmpeg_paths ${_ffmpeg_paths} --extra-cflags=-I${dir})
      endforeach()
    else()
      set(_ffmpeg_paths ${_ffmpeg_paths} --extra-cflags=-I${${uname}_INCLUDE_DIR})
    endif()
    if(TARGET ${confname})
      set(_ffmpeg_deps ${_ffmpeg_deps} ${confname}) #External project targets must correspond to confname's
    endif()
    if(${uname}_LIBRARIES)
      set(FFMPEG_DEP_LIBRARIES ${FFMPEG_DEP_LIBRARIES} ${${uname}_LIBRARIES})
    else()
      set(FFMPEG_DEP_LIBRARIES ${FFMPEG_DEP_LIBRARIES} ${${uname}_LIBRARY})
    endif()
  else()
      set(_ffmpeg_conf ${_ffmpeg_conf} --disable-${confname})
  endif()
endmacro(_ffmpeg_maybe_add)

_ffmpeg_maybe_add(ZLIB   zlib      "A Massively Spiffy Yet Delicately Unobtrusive Compression Library" http://zlib.net)
_ffmpeg_maybe_add(BZIP2  bzlib     "A freely available, patent free, high-quality data compressor."    http://www.bzip.org)
_ffmpeg_maybe_add(x264   libx264   "A free library for encoding videos streams into the H.264/MPEG-4 AVC format" http://www.videolab.org/developers/x264.html)
_ffmpeg_maybe_add(theora libtheora "Video compression for the OGG format from Xiph.org" http://www.theora.org)
_ffmpeg_maybe_add(va     vaapi     "Enables hardware accelerated video decode/encode for prevailing standard formats." http://www.freedesktop.org/wiki/Software/vaapi)
_ffmpeg_maybe_add(VPX    libvpx    "An open, royalty-free, media file format for the web." http://www.webmproject.org/code)

## I haven't quite figured out xvid's build yet...It's hard to change the install prefix
## using their build system.
#_ffmpeg_maybe_add(xvid   libxvid   "The XVID video codec." http://www.xvid.org)

#if(X264_FOUND) #X264 requires gpl
#  set(_ffmpeg_conf ${_ffmpeg_conf} --enable-gpl)
#endif()

## Finish up - must be after _ffmpeg_maybe_add section
#  Add library paths for each library to config's cflags
foreach(lib ${FFMPEG_DEP_LIBRARIES})
    get_filename_component(dir ${lib} PATH)
    set(_ffmpeg_paths ${_ffmpeg_paths} --extra-ldflags=-L${dir})
endforeach()

##
## EXTERNAL PROJECT CALL
##

show(_ffmpeg_conf)
foreach(e ${_ffmpeg_paths})
  show(e)
endforeach()
include(ExternalProject)
ExternalProject_Add(ffmpeg
  DEPENDS           ${_ffmpeg_deps} yasm
  GIT_REPOSITORY    ${FFMPEG_GIT_REPOSITORY}
  GIT_TAG           ${FFMPEG_GIT_BRANCH}
  CONFIGURE_COMMAND 
        <SOURCE_DIR>/configure
          --prefix=<INSTALL_DIR>
          --yasmexe=${YASM_ROOT_DIR}/bin/yasm
          --enable-shared
          --enable-gpl
          --enable-version3
          --extra-cflags=-g
          ${_ffmpeg_conf}
          ${_ffmpeg_paths}
  BUILD_IN_SOURCE TRUE # hard to get ffmpeg not to do this ... maybe forces rebuilds T.T
)
get_target_property(FFMPEG_ROOT_DIR ffmpeg _EP_INSTALL_DIR)

set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT_DIR}/include CACHE PATH "Location of FFMPEG headers.")
macro(FFMPEG_FIND name)
  set(FFMPEG_${name}_LIBRARY
    ${FFMPEG_ROOT_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    CACHE PATH "Location of lib${name} library." FORCE)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${FFMPEG_${name}_LIBRARY})
endmacro()
FFMPEG_FIND(avformat)
FFMPEG_FIND(avdevice)
FFMPEG_FIND(avcodec)
FFMPEG_FIND(avutil)
FFMPEG_FIND(swscale)

##
## OUTPUT
##

set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${FFMPEG_DEP_LIBRARIES})
PRINT_ENABLED_FEATURES()
PRINT_DISABLED_FEATURES()
set(FFMPEG_FOUND 1)
