cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
project(CudaTransformer VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(/usr/local/cuda/include)
include_directories(/usr/include/c++/11)
include_directories(/usr/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE CUDA_SOURCES
    "src/*.cu"
)

add_executable(cuda_transformer ${CUDA_SOURCES})

set_target_properties(cuda_transformer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Limpieza opcional
add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/bin
)