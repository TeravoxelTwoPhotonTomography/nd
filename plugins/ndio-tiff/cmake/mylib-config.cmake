# Locate mylib library  

include(ExternalProject)
include(FindPackageHandleStandardArgs)

set(MYLIB_GIT_REPOSITORY git@bitbucket.org:nclack/mylib.git CACHE STRING "Location of the git repository for libnd.")
if(NOT TARGET libmylib)
  ExternalProject_Add(libmylib
      GIT_REPOSITORY ${MYLIB_GIT_REPOSITORY}      
      UPDATE_COMMAND git pull origin master
      CMAKE_ARGS     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )
endif()

ExternalProject_Get_Property(libmylib INSTALL_DIR)
set(MYLIB_INCLUDE_DIR ${INSTALL_DIR}/include/mylib)
set(MYLIB_LIBRARY   mylib)
set(MYLIB_LIBRARIES mylib mytiff myfft)

foreach(tgt ${MYLIB_LIBRARIES})
  add_library(${tgt} IMPORTED STATIC)
  add_dependencies(${tgt} libmylib)
  set_target_properties(${tgt} PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION  ${INSTALL_DIR}/lib/mylib/${CMAKE_STATIC_LIBRARY_PREFIX}${tgt}${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
endforeach()

find_package_handle_standard_args(MYLIB DEFAULT_MSG
  MYLIB_INCLUDE_DIR
  MYLIB_LIBRARIES
)
