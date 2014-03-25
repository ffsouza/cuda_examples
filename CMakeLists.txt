cmake_minimum_required(VERSION 2.8)
project(cuda_exemples)

find_package(CUDA)
include(FindCUDA)

CUDA_ADD_EXECUTABLE(cuda_info cuda_info.cu)
CUDA_ADD_EXECUTABLE(vector_sum vector_sum.cu)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#TARGET_LINK_LIBRARIES(lbmslv /usr/local/cuda/lib64/libcudart.so)
