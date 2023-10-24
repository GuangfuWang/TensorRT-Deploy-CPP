set(FIGHT_SRC
        ${PROJECT_SOURCE_DIR}/src/trt_deploy.cpp
        ${PROJECT_SOURCE_DIR}/src/util.cpp
        ${PROJECT_SOURCE_DIR}/src/config.cpp
        ${PROJECT_SOURCE_DIR}/src/preprocessor.cpp
        ${PROJECT_SOURCE_DIR}/src/preprocess_ops.cpp
        ${PROJECT_SOURCE_DIR}/src/trt_deployresult.cpp
        ${PROJECT_SOURCE_DIR}/src/postprocessor.cpp
        ${PROJECT_SOURCE_DIR}/src/model.cpp
        )


set(FIGHT_HEADER
        ${PROJECT_SOURCE_DIR}/src/trt_deploy.h
        ${PROJECT_SOURCE_DIR}/src/util.h
        ${PROJECT_SOURCE_DIR}/src/macro.h
        ${PROJECT_SOURCE_DIR}/src/cmdline.h
        ${PROJECT_SOURCE_DIR}/src/config.h
        ${PROJECT_SOURCE_DIR}/src/preprocessor.h
        ${PROJECT_SOURCE_DIR}/src/preprocess_util.hpp
        ${PROJECT_SOURCE_DIR}/src/preprocess_ops.h
        ${PROJECT_SOURCE_DIR}/src/trt_deployresult.h
        ${PROJECT_SOURCE_DIR}/src/postprocessor.h
        ${PROJECT_SOURCE_DIR}/src/model.h
        )

set(FIGHT_MAIN
        ${PROJECT_SOURCE_DIR}/src/main.cpp
        )
set(FIGHT_TEST_MAIN
        ${PROJECT_SOURCE_DIR}/test/main_test.cpp
        )

