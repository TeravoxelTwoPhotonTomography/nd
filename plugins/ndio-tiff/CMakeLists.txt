cmake_minimum_required(VERSION 2.8)
project(ndio-tiff-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################

find_package(mylib CONFIG PATHS cmake)
if(MYLIB_FOUND)
  include_directories(${MYLIB_INCLUDE_DIR})

  get_directory_property(ND_SOURCE_DIR PARENT_DIRECTORY)

  include_directories(${ND_SOURCE_DIR}) # from parent scope

  ################################################################################
  # SOURCE
  ################################################################################

  file(GLOB SRCS src/*.c)
  file(GLOB HDRS src/*.h)

  ################################################################################
  # TARGETS
  ################################################################################

  add_library(ndio-tiff MODULE ${SRCS} ${HDRS})
  target_link_libraries(ndio-tiff ${MYLIB_LIBRARIES} nd)
endif(MYLIB_FOUND)

###############################################################################
#  Documentation
###############################################################################

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
                 ${PROJECT_BINARY_DIR}/doc/Doxyfile
                 @ONLY
                 )
  add_custom_target(ndio-tiff-docs
    ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT           "Generating Docs for Tiff IO Plugin"
    VERBATIM
    )
endif()

###############################################################################
#  Install
###############################################################################
install(TARGETS ndio-tiff
#  EXPORT ndio-tiff-targets
  LIBRARY DESTINATION bin/${NDIO_PLUGIN_PATH}
)
