cmake_minimum_required(VERSION 3.20)


find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)


add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

project(tinyfuseml LANGUAGES CXX CUDA)

# Require GCC 11 or newer for CUDA 11.8 compatibility
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "GCC/G++ 11 or newer is required")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(DEFAULT_ARCHS "61;70;75;80;86;89;90;90-virtual")
    set(CMAKE_CUDA_ARCHITECTURES "${DEFAULT_ARCHS}" CACHE STRING "CUDA architecture list (Compute Capability)")
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS_WRITE_INCLUDE_PATHS_DIRECTLY TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tinyfuseml
    src/main.cu
    src/mlp_kernel.cu
    src/image_loader.cpp
    src/image_writer.cpp
    src/fourier_encoding.cpp
    src/hash_encoding.cu
)

target_include_directories(tinyfuseml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# This is a workaround to force the -I flag into compile_commands.json
target_compile_options(tinyfuseml PRIVATE
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(tinyfuseml PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

set_target_properties(tinyfuseml PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(tinyfuseml PRIVATE CUDA::cudart)
target_link_libraries(tinyfuseml PRIVATE ${OpenCV_LIBS})




