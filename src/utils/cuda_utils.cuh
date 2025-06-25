// cuda_utils.cuh
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) 
                  << " in file " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

template <typename T>
void cudaMallocWrapper(T** ptr, size_t size) {
    CHECK_CUDA_ERROR(cudaMalloc((void**)ptr, size * sizeof(T)));
}

template <typename T>
void cudaMemcpyWrapper(T* dst, const T* src, size_t size, cudaMemcpyKind kind) {
    CHECK_CUDA_ERROR(cudaMemcpy(dst, src, size * sizeof(T), kind));
}

template <typename T>
void cudaFreeWrapper(T* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

#endif // CUDA_UTILS_H