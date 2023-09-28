find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    message(STATUS "Found Cuda with version: ${CUDA_VERSION}")
endif ()

set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(YAML-CPP REQUIRED)


set(FIGHT_LIBS ${OpenCV_LIBS} yaml-cpp cudart nvinfer nvinfer_plugin)