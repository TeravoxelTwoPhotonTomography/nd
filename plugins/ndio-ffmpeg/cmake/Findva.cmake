# Locate libva libraries
# This module defines
# VAAPI_LIBRARY, the name of the library to link against
# VAAPI_FOUND, if false, do not try to link
# VAAPI_INCLUDE_DIR, where to find header
#
find_path(VAAPI_INCLUDE_DIR va/va.h)
find_library(VAAPI_LIBRARY NAME va)
if(NOT VAAPI_LIBRARY)
  ## Building as an external project requires autotools
  ## FIXME: Don't have the write test system at the moment
  ## FIXME: Add ExternalProject
# include(ExternalProject)
# ExternalProject_Add(vaapi
#   GIT_REPOSITORY git://anongit.freedesktop.org/git/libva
#   CONFIGURE_COMMAND <SOURCE_DIR>/configure
#         --prefix=<INSTALL_DIR>
# )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VAAPI DEFAULT_MSG
  VAAPI_LIBRARY
  VAAPI_INCLUDE_DIR
)
