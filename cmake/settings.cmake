## Author: Guangfu WANG.
## Date: 2023-08-20.
#set cpp version used in this project.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#this is equivalently to -fPIC in cxx_flags.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#define options for custom build targets.
option(FIGHT_TEST "Build fight test program." ON)
option(FIGHT_PREPROCESS_GPU "Use GPU version of preprocessing pipeline" ON)
set(FIGHT_INPUT_NAME "image" CACHE STRING "Input layer name for tensorrt deploy.")
set(FIGHT_OUTPUT_NAMES "scores" CACHE STRING "Output layer names for tensorrt deploy, seperated with comma or colon")
set(FIGHT_DEPLOY_MODEL "weight/fight/fight_pptsm.engine" CACHE STRING "Used deploy AI model file (/path/to/*.engine)")

# generate config.h in src folder.
configure_file(
        "${PROJECT_SOURCE_DIR}/src/macro.h.in"
        "${PROJECT_SOURCE_DIR}/src/macro.h"
        @ONLY
)

set(FIGHT_DEPLOY_LIB_NAME "fight_deploy_lib")
set(FIGHT_DEPLOY_MAIN_NAME "fight_pptsm_tensorrt")
set(FIGHT_DEPLOY_TEST_MAIN_NAME "fight_pptsm_trt_test")

set(CMAKE_INSTALL_RPATH "\$ORIGIN")
set(CMAKE_INSTALL_PREFIX "install")
add_link_options("-Wl,--as-needed")
