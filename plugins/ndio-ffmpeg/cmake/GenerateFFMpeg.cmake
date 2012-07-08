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
  if(not ${VAR})
    set(${VAR} ${VAL})
  endif()
endmacro(_ffmpeg_setdefault)

_ffmpeg_setdefault(FFMPEG_GIT_REPOSITORY git://github.com/FFmpeg/FFmpeg.git)
_ffmpeg_setdefault(FFMPEG_GIT_BRANCH     release/0.11.1)

show(FFMPEG_GIT_REPOSITORY)
show(FFMPEG_GIT_BRANCH)

##
## OPTIONAL DEPENDENCIES
##

# Modifies FFMPEG_DEPENDENCIES and _ffmpeg_conf
# Adds libraries and configure lines if a library is found.
macro(_ffmpeg_maybe_add name confname)
  find_package(${name} OPTIONAL)
  if(${name}_FOUND)
    set(_ffmpeg_conf ${_ffmpeg_conf} --enable-${confname})
    if(${name}_LIBRARIES)
      set(FFMPEG_DEPENDENCIES ${FFMPEG_DEPENDENCIES} ${name}_LIBRARIES)
    else()
      set(FFMPEG_DEPENDENCIES ${FFMPEG_DEPENDENCIES} ${name}_LIBRARY)
    endif()
  else()
    set(_ffmpeg_conf ${_ffmpeg_conf} --disable-${confname})
  endif()
endmacro(_ffmpeg_maybe_add)

_ffmpeg_maybe_add(ZLIB   zlib)
_ffmpeg_maybe_add(BZip2  bzlib)
_ffmpeg_maybe_add(Vorbis libvorbis)
_ffmpeg_maybe_add(x264   libx264)
_ffmpeg_maybe_add(theora libtheora)
_ffmpeg_maybe_add(va     vaapi)
_ffmpeg_maybe_add(VPX    libvpx)

##
## REQUIRED DEPENDENCIES
##

ExternalProject_Add(yasm
  URL               http://www.tortall.net/projects/yasm/releases/yasm-1.2.0.tar.gz
  URL_MD5           4cfc0686cf5350dd1305c4d905eb55a6
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
  )
get_target_property(YASM_ROOT_DIR yasm _EP_INSTALL_DIR)

if(APPLE) #FFMPEG may rely on these Frameworks
  find_library(CF_LIBS  CoreFoundation)
  find_library(VDA_LIBS VideoDecodeAcceleration)
  find_library(CV_LIBSa CoreVideo)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${CF_LIBS} ${VDA_LIBS} ${CV_LIBS})
endif()

##
## EXTERNAL PROJECT CALL
##

include(ExternalProject)
ExternalProject_Add(ffmpeg
  GIT_REPOSITORY    ${FFMPEG_GIT_REPOSITORY}
  GIT_TAG           ${FFMPEG_GIT_BRANCH}
  CONFIGURE_COMMAND <SOURCE_DIR>/configure
        --prefix=<INSTALL_DIR>
        --yasmeexe=${YASM_ROOT_DIR}/bin/yasm
        --enable-shared
        --extra-cflags="-g"
        ${_ffmpeg_conf}
  BUILD_IN_SOURCE TRUE # hard to get ffmpeg not to do this ... maybe forces rebuilds T.T
)
get_target_property(FFMPEG_ROOT_DIR ffmpeg _EP_INSTALL_DIR)

set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT_DIR}/include CACHE PATH "Location of FFMPEG headers.")
macro(FFMPEG_FIND name)
  set(FFMPEG_${name}_LIBRARY
    #${FFMPEG_ROOT_DIR}/lib/lib${name}/${CMAKE_SHARED_LIBRARY_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${FFMPEG_ROOT_DIR}/lib/lib${name}/${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    CACHE PATH "Location of lib${name} library.")
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

set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${FFMPEG_DEPENDENCIES})
set(FFMPEG_FOUND 1)
