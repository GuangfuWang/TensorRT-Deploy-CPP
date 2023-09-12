# if you use vcpkg and compile on windows, you can specify the relevant macro here.
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    # embedded platform with linux os.
    set(CUDA_BASE "/usr/local/cuda/targets/aarch64-linux")
else ()
    # normal platform with linux.
    set(CUDA_BASE "/usr/local/cuda")
endif ()

set(CMAKE_CUDA_COMPILER "${CUDA_BASE}/bin/nvcc")