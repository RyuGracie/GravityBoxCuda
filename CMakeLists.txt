cmake_minimum_required(VERSION 3.18)
project(GravityBoxCUDA LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cu
    src/physics.cu
    src/renderer.cpp
)

# Headers
set(HEADERS
    include/particle.h
    include/physics.cuh
    include/renderer.h
)

# Create executable
add_executable(particles ${SOURCES} ${HEADERS})

# Set CUDA architectures (adjust based on your GPU)
# Common values: 50, 52, 60, 61, 70, 75, 80, 86, 89, 90
set_target_properties(particles PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86;89"  # Adjust for your GPU
)

# Link libraries
target_link_libraries(particles
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    CUDA::cudart
    CUDA::cuda_driver
)

# Compiler flags for better optimization
target_compile_options(particles PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >
)

# Print configuration info
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")