include(ExternalProject)
include(FindPackageHandleStandardArgs)

if(NOT TARGET gtest)
  ## DOWNLOAD AND BUILD
  #  Must use the SVN because cmake can't handle the zip.  If only there were a tar.gz!
  ExternalProject_Add(gtest
    SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk/
    #URL http://code.google.com/p/googletest/downloads/detail?name=gtest-1.6.0.zip
    #UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
               -DBUILD_SHARED_LIBS:BOOL=TRUE
    )
endif()

get_target_property(GTEST_SRC_DIR  gtest _EP_SOURCE_DIR)
get_target_property(GTEST_ROOT_DIR gtest _EP_BINARY_DIR)
show(GTEST_ROOT_DIR)
if(NOT TARGET libgtest)
  add_library(libgtest      SHARED IMPORTED )
  add_library(libgtest-main SHARED IMPORTED )
  set_target_properties(libgtest PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_IMPLIB           "${GTEST_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}"
    IMPORTED_LOCATION         "${GTEST_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}gtest${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
  set_target_properties(libgtest-main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_IMPLIB           "${GTEST_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}"
    IMPORTED_LOCATION         "${GTEST_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}gtest_main${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
endif()

macro(gtest_copy_shared_libraries _target)  
  foreach(_lib libgtest libgtest-main)
    get_target_property(_name ${_lib}    IMPORTED_LOCATION)
    #show(_name)
    add_custom_command(TARGET ${_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          ${_name}
          $<TARGET_FILE_DIR:${_target}>
          )  
  endforeach()
endmacro()

#set_property(TARGET libgtest PROPERTY IMPORTED_LOCATION ${GTEST_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
#set_property(TARGET libgtest-main PROPERTY IMPORTED_LOCATION ${GTEST_ROOT_DIR}/${CMAKE_CFG_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})

set(GTEST_LIBRARY libgtest)
set(GTEST_MAIN_LIBRARY libgtest-main)
set(GTEST_BOTH_LIBRARIES libgtest libgtest-main)
set(GTEST_INCLUDE_DIR ${GTEST_SRC_DIR}/include)

find_package_handle_standard_args(GTEST DEFAULT_MSG
  GTEST_BOTH_LIBRARIES
  GTEST_INCLUDE_DIR
)
