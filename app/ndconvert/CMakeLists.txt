set(plugins
  ndio-ffmpeg
  ndio-tiff
  ndio-raw
  ndio-series
  ndio-hdf5)
find_package(ND COMPONENTS ${plugins} PATHS ${PROJECT_BINARY_DIR}) # for debugging mostly
add_executable(ndconvert ndconvert.c)
target_link_libraries(ndconvert ${ND_LIBRARIES})
nd_copy_plugins_to_target(ndconvert ${plugins})
install(TARGETS ndconvert RUNTIME DESTINATION bin)
