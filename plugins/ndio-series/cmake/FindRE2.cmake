# Would be nice if the source code was downloadable from an
# archive.  Otherwise, this requires hg to be installed.
#
# May have to look for a msvc compatible source tree elsewhere
# see: https://bitbucket.org/root_op/re2-msvc
include(ExternalProject)
include(FindPackageHandleStandardArgs)
if(NOT TARGET re2)
  ExternalProject_Add(re2
    #    DOWNLOAD_COMMAND hg clone https://re2.googlecode.com/hg re2
    URL     http://dl.dropbox.com/u/782372/Software/re2.zip
    URL_MD5 7f98f65c958248d85f6270c96373afb9
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    )
endif()
get_target_property(RE2_SRC_DIR re2 _EP_SOURCE_DIR)
get_target_property(RE2_ROOT_DIR re2 _EP_BINARY_DIR)

add_library(libre2 IMPORTED STATIC)
add_dependencies(libre2 re2)
set_property(TARGET libre2 PROPERTY IMPORTED_LOCATION ${RE2_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}re2${CMAKE_STATIC_LIBRARY_SUFFIX})

set(RE2_LIBRARY libre2)
set(RE2_INCLUDE_DIRS ${RE2_SRC_DIR})

find_package_handle_standard_args(RE2 DEFAULT_MSG
	RE2_LIBRARY
	RE2_INCLUDE_DIRS
)
