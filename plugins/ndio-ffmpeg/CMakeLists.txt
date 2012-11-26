cmake_minimum_required(VERSION 2.8)
project(ndio-ffmpeg-plugin)

macro(show v)
  message("${v} is ${${v}}")
endmacro()
macro(showeach v)
  message("${v} is:")
  foreach(e  ${${v}})
    message("   ${e}")
  endforeach()
endmacro()
macro(show_target_prop t p)
  get_target_property(pp ${t} ${p})
  message("The ${p} of ${t} is ${pp}")
endmacro()

################################################################################
# CONFIG
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(ROOT_3RDPARTY_DIR ${PROJECT_SOURCE_DIR})

find_package(FFMPEG CONFIG PATHS cmake)
if(FFMPEG_FOUND)
  include_directories(${FFMPEG_INCLUDE_DIR})

  get_directory_property(ND_SOURCE_DIR PARENT_DIRECTORY)

  include_directories(${ND_SOURCE_DIR}) # from parent scope

  ##############################################################################
  # SOURCE
  ##############################################################################

  file(GLOB SRCS src/*.c)
  file(GLOB HDRS src/*.h)

  ##############################################################################
  # TARGETS
  ##############################################################################

  add_definitions(-fPIC)
  add_library(ndio-ffmpeg MODULE ${SRCS} ${HDRS})
  target_link_libraries(ndio-ffmpeg ${FFMPEG_LIBRARIES} nd)
#showeach(FFMPEG_LIBRARIES)
  set_target_properties(ndio-ffmpeg PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_dependencies(ndio-ffmpeg ffmpeg)

  ## Copy FFMPEG shared libs to plugin build dir so build functions in place
  #  (Not super sure what to do for non MSVC)
  if(MSVC)
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
      file(COPY ${FFMPEG_SHARED_LIBS} DESTINATION "${PROJECT_BINARY_DIR}/${cfg}") 
    endforeach(cfg)
  else()
    showeach(FFMPEG_SHARED_LIBS)
    foreach(e ${FFMPEG_SHARED_LIBS})
      show_target_prop(${e} LOCATION)
      #get_target_property(pp ${e} LOCATION)
      #add_library(lib${e} SHARED IMPORTED)
      #set_target_properties(lib${e} PROPERTIES IMPORTED_LOCATION ${pp})
      #install(TARGETS lib${e} DESTINATION bin/${NDIO_PLUGIN_PATH}) 
    endforeach ()
    #install(TARGETS ${FFMPEG_SHARED_LIBS} DESTINATION bin/${NDIO_PLUGIN_PATH})
    #install(EXPORT ffmpeg_shared_lib_targets DESTINATION bin/plugins)
    #file(COPY ${FFMPEG_SHARED_LIBS} DESTINATION ${PROJECT_BINARY_DIR})
  endif()

endif(FFMPEG_FOUND)

################################################################################
#  Documentation
################################################################################

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
                 ${PROJECT_BINARY_DIR}/doc/Doxyfile
                 @ONLY
                 )
  add_custom_target(ndio-ffmpeg-docs
    ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT           "Generating Docs for FFMPEG IO Plugin"
    VERBATIM
    )
endif()

################################################################################
#  Install
################################################################################
install(TARGETS ndio-ffmpeg
  EXPORT ndio-ffmpeg-targets
  RUNTIME DESTINATION bin/${NDIO_PLUGIN_PATH}
  LIBRARY DESTINATION bin/${NDIO_PLUGIN_PATH}
)
#export(TARGETS ndio-ffmpeg FILE ndio-ffmpeg-targets.cmake)
#install(EXPORT ndio-ffmpeg-targets DESTINATION cmake)
