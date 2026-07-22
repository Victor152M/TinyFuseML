cmake_minimum_required(VERSION 3.20)

project(tinyfuseml LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")


# Require GCC 11 or newer for CUDA 11.8 compatibility
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "GCC/G++ 11 or newer is required")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_EXPORT_COMPILE_COMMANDS_WRITE_INCLUDE_PATHS_DIRECTLY TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_executable(tinyfuseml
    src/main.cu
    src/mlp_kernel.cu
    src/image_loader.cpp
    src/image_writer.cpp
    src/hash_encoding.cu
    src/volumetric_renderer.cu
    src/trainer.cu
    src/cli.cpp
    src/sdf_loader.cpp
)

target_compile_definitions(tinyfuseml PRIVATE DEBUG_PRINT)

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




