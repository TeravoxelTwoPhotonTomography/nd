# Include this to generate and build FFMPEG
# INPUT
#   FFMPEG_GIT_REPOSITORY     has default
#   FFMPEG_GIT_BRANCH         has deefalt
#
# OUTPUT
#   FFMPEG_FOUND
#   FFMPEG_LIBRARIES
#   FFMPEG_INCLUDE_DIR

include(GenerateFFMPEG)
include(FindPackageHandleStandardArgs)

GenerateFFMPEG(https://github.com/FFmpeg/FFmpeg.git n0.11.1)
find_package_handle_standard_args(FFMPEG DEFAULT_MSG
  FFMPEG_LIBRARIES
  FFMPEG_INCLUDE_DIR
)
