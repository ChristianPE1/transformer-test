// common.cuh
#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CURAND(call) { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        std::cerr << "CURAND error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Utility functions
__host__ __device__ inline float relu(float x) {
    return fmaxf(0.0f, x);
}

__host__ __device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#endif // COMMON_CUH