// src/training/optimizer.cu
#include "optimizer.cuh"
#include "utils/cuda_utils.cuh"

__global__ void updateWeightsKernel(float *weights, float *gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

Optimizer::Optimizer(float learning_rate) : learning_rate(learning_rate) {}

SGD::SGD(float learning_rate) : Optimizer(learning_rate) {
    // Constructor implementation
    
}

void SGD::step(float* params, float* grads, size_t size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateWeightsKernel<<<numBlocks, blockSize>>>(params, grads, learning_rate, size);
    cudaDeviceSynchronize();
}
