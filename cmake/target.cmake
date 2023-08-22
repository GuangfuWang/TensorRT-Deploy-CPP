add_library(${FIGHT_DEPLOY_LIB_NAME} ${FIGHT_SRC}
        ../src/postprocessor.cpp
        ../src/postprocessor.h)
target_include_directories(${FIGHT_DEPLOY_LIB_NAME} PUBLIC ${CUDA_INCLUDE_DIR})
target_link_libraries(${FIGHT_DEPLOY_LIB_NAME} PUBLIC ${FIGHT_LIBS})

add_executable(${FIGHT_DEPLOY_MAIN_NAME} ${FIGHT_HEADER} ${FIGHT_MAIN})
target_link_libraries(${FIGHT_DEPLOY_MAIN_NAME} PUBLIC ${FIGHT_LIBS} ${FIGHT_DEPLOY_LIB_NAME})

if (FIGHT_TEST)
    message(STATUS "Build Test...")
    add_executable(${FIGHT_DEPLOY_TEST_MAIN_NAME} ${FIGHT_HEADER} ${FIGHT_TEST_MAIN})
    target_link_libraries(${FIGHT_DEPLOY_TEST_MAIN_NAME} PRIVATE ${FIGHT_LIBS} ${FIGHT_DEPLOY_LIB_NAME})
endif ()

