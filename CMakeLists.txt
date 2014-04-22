cmake_minimum_required(VERSION 2.8)
project(cuda_exemples)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_scripts/  ${CMAKE_MODULE_PATH})

find_package(CUDA)
include(FindCUDA)

find_package ( SOIL REQUIRED )
include_directories(${SOIL_INCLUDE_DIR})

# OpenGL
find_package(OpenGL)
include_directories(${OPENGL_INCLUDE_DIR})

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_PATH})

# Glut
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_PATH})

set (LIBRARIES ${SOIL_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${CUDA_LIBRARIES})
#set (LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${SOIL_LIBRARIES} ${SFML_LIBRARIES} ${GLM_LIBRARIES})

CUDA_ADD_EXECUTABLE(cuda_info cuda_info.cu)

CUDA_ADD_EXECUTABLE(vector_sum vector_sum.cu)

CUDA_ADD_EXECUTABLE(vector_sum_thread vector_sum_thread.cu)

CUDA_ADD_EXECUTABLE(julia_set julia_set.cu)
TARGET_LINK_LIBRARIES(julia_set ${LIBRARIES})

CUDA_ADD_EXECUTABLE(ripple_animation ripple_animation.cu)
TARGET_LINK_LIBRARIES(ripple_animation ${LIBRARIES})

CUDA_ADD_EXECUTABLE(dot_product_shared_memory dot_product_shared_memory.cu)

CUDA_ADD_EXECUTABLE(image_shared_memory image_shared_memory.cu)
TARGET_LINK_LIBRARIES(image_shared_memory ${LIBRARIES})

CUDA_ADD_EXECUTABLE(raytracing_constant_memory raytracing_constant_memory.cu)
TARGET_LINK_LIBRARIES(raytracing_constant_memory ${LIBRARIES})

CUDA_ADD_EXECUTABLE(heat_transfer_texture_memory heat_transfer_texture_memory.cu)
TARGET_LINK_LIBRARIES(heat_transfer_texture_memory ${LIBRARIES})

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall -pthread")
list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#TARGET_LINK_LIBRARIES(lbmslv /usr/local/cuda/lib64/libcudart.so)
