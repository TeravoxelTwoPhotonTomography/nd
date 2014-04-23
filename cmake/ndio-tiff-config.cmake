# LIBND PLUGIN
#

include(ExternalProject)
include(FindPackageHandleStandardArgs)

# SANITIZE LISTS
# allows us to pass the list in through the cmake command line
string(REPLACE ";" "^^" SEP_ND_LIBRARIES "${ND_LIBRARIES}")  
string(REPLACE ";" "^^" SEP_NDIO_PLUGIN_PATH "${NDIO_LIBRARIES}")
string(REPLACE ";" "^^" SEP_GTEST_BOTH_LIBRARIES "${GTEST_BOTH_LIBRARIES}")

set(NDIO_TIFF_GIT_REPOSITORY ssh://git@bitbucket.org/nclack/ndio-tiff.git CACHE STRING "Location of the git repository for ndio-tiff.")
ExternalProject_Add(ndio-tiff-plugin
  GIT_REPOSITORY ${NDIO_TIFF_GIT_REPOSITORY}
  LIST_SEPARATOR ^^
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
             -DND_LIBRARIES:STRING=${SEP_ND_LIBRARIES}
             -DND_INCLUDE_DIRS:STRING=${ND_INCLUDE_DIRS}
             -DNDIO_PLUGIN_PATH:STRING=${NDIO_PLUGIN_PATH}
             -DGTEST_INCLUDE_DIR:PATH=${GTEST_INCLUDE_DIR}
             -DGTEST_BOTH_LIBRARIES:STRING=${SEP_GTEST_BOTH_LIBRARIES}
  )
ExternalProject_Get_Property(ndio-tiff-plugin INSTALL_DIR)

add_library(ndio-tiff MODULE IMPORTED)
set(plugin ndio-tiff)
set_target_properties(${plugin} PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "C"
  IMPORTED_LOCATION "${INSTALL_DIR}/bin/plugins/${CMAKE_SHARED_MODULE_PREFIX}${plugin}${CMAKE_SHARED_MODULE_SUFFIX}"
)
add_dependencies(ndio-tiff libnd)

### install plugins to parent project plugins
get_target_property(_path ndio-tiff IMPORTED_LOCATION)
install(FILES ${_path} DESTINATION bin/plugins)

### exit variables
set(ND_ndio-tiff_FOUND TRUE)
