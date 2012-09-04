cmake_minimum_required(VERSION 2.8)
project(ndio-hdf5-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()

################################################################################
# CONFIG
################################################################################
set(NDIO_HDF5_TEST_DATA_PATH ${PROJECT_SOURCE_DIR}/test/data)
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
include_directories(${PROJECT_BINARY_DIR})

find_package(HDF5 CONFIG PATHS cmake CONFIGS HDF5Config.cmake)
if(HDF5_FOUND)
  include_directories(${HDF5_INCLUDE_DIRS})
  get_directory_property(ND_SOURCE_DIR PARENT_DIRECTORY)
  include_directories(${ND_SOURCE_DIR})

  ##############################################################################
  # SOURCE
  ##############################################################################

  file(GLOB SRCS src/*.c)
  file(GLOB HDRS src/*.h)

  ##############################################################################
  # TARGETS
  ##############################################################################

  add_definitions(-fPIC)
  add_library(ndio-hdf5 MODULE ${SRCS} ${HDRS})
  target_link_libraries(ndio-hdf5 nd ${HDF5_LIBRARIES})

  ##############################################################################
  #  Testing
  ##############################################################################

  find_package(GTEST CONFIG PATHS cmake NAMES GTest)

  file(GLOB TEST_SOURCES test/*.cc)
  if(GTEST_FOUND AND TEST_SOURCES)
    find_package(Threads)
    enable_testing()
    include_directories(${PROJECT_SOURCE_DIR})
    include_directories(${GTEST_INCLUDE_DIR})
    add_executable(ndio-hdf5-tests ${TEST_SOURCES} config.h)
    target_link_libraries(ndio-hdf5-tests
      ${GTEST_BOTH_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT}
      nd
      )
    add_dependencies(ndio-hdf5-tests gtest)
    gtest_copy_shared_libraries(ndio-hdf5-tests)
    install(TARGETS ndio-hdf5-tests DESTINATION bin)
    add_test(TestNDIOSeries ndio-hdf5-tests)
  endif()

  ##############################################################################
  #  Install
  ##############################################################################
  install(TARGETS ndio-hdf5
    EXPORT ndio-hdf5-targets
    LIBRARY DESTINATION bin/${NDIO_PLUGIN_PATH}
  )
  export(TARGETS ndio-hdf5 FILE ndio-hdf5-targets.cmake)
  install(EXPORT ndio-hdf5-targets DESTINATION cmake)

endif(HDF5_FOUND)

################################################################################
#  Documentation
################################################################################

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
                 @ONLY
                 )
  add_custom_target(ndio-hdf5-docs
    ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT           "Generating API Docs"
    VERBATIM
    )
endif()