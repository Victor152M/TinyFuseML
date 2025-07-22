cmake_minimum_required(VERSION 3.18)
project(cuda_project LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Set the host compiler before enabling CUDA
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_STANDARD 14)

# Specify your GPU architecture (GTX 1080 = sm_61)
set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_EXPORT_COMPILE_COMMANDS_WRITE_INCLUDE_PATHS_DIRECTLY TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda_project
    src/main.cu
    src/mlp_kernel.cu
)

target_include_directories(cuda_project PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# This is a workaround to force the -I flag into compile_commands.json
target_compile_options(cuda_project PRIVATE
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(cuda_project PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda_project PRIVATE CUDA::cudart)

