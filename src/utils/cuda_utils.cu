// cuda_utils.cu
#include "utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void* allocateCudaMemory(size_t size) {
    void* d_ptr;
    checkCudaError(cudaMalloc(&d_ptr, size), "Failed to allocate CUDA memory");
    return d_ptr;
}

void copyToDevice(void* d_ptr, const void* h_ptr, size_t size) {
    checkCudaError(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice), "Failed to copy to device");
}

void copyToHost(void* h_ptr, const void* d_ptr, size_t size) {
    checkCudaError(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost), "Failed to copy to host");
}

void freeCudaMemory(void* d_ptr) {
    checkCudaError(cudaFree(d_ptr), "Failed to free CUDA memory");
}

void synchronizeCuda() {
    checkCudaError(cudaDeviceSynchronize(), "CUDA device synchronization failed");
}