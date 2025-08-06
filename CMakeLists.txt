cmake_minimum_required(VERSION 3.18)

set(CUDAToolkit_ROOT "/usr/local/cuda-11.8")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Set the host compiler before enabling CUDA
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Specify your GPU architecture (GTX 1080 = sm_61)
set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_EXPORT_COMPILE_COMMANDS_WRITE_INCLUDE_PATHS_DIRECTLY TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda_project
    src/main.cu
    src/mlp_kernel.cu
    src/image_loader.cpp
    src/image_writer.cpp
    src/fourier_encoding.cpp
    src/hash_encoding.cu
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
    ${OpenCV_INCLUDE_DIRS}
)

set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda_project PRIVATE CUDA::cudart)
target_link_libraries(cuda_project PRIVATE ${OpenCV_LIBS})




