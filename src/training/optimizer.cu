// src/training/optimizer.cu
#include "optimizer.cuh"
#include "utils/cuda_utils.cuh"
#include <algorithm>

__global__ void clipGradientsKernel(float *gradients, float max_norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = gradients[idx];
        if (grad > max_norm) gradients[idx] = max_norm;
        else if (grad < -max_norm) gradients[idx] = -max_norm;
    }
}

__global__ void updateWeightsWithMomentumKernel(float *weights, float *gradients, float *momentum, 
                                               float learning_rate, float momentum_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update momentum: v = momentum_factor * v + learning_rate * grad
        momentum[idx] = momentum_factor * momentum[idx] + learning_rate * gradients[idx];
        // Update weights: w = w - v
        weights[idx] -= momentum[idx];
    }
}

__global__ void updateWeightsKernel(float *weights, float *gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

Optimizer::Optimizer(float learning_rate) : learning_rate(learning_rate) {}

SGD::SGD(float learning_rate, float momentum_factor) 
    : Optimizer(learning_rate), momentum_factor(momentum_factor), momentum_buffer(nullptr), buffer_size(0) {
    
}

SGD::~SGD() {
    if (momentum_buffer) {
        cudaFree(momentum_buffer);
    }
}

void SGD::step(float* params, float* grads, size_t size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // TEMPORARILY DISABLE gradient clipping to see if it's the problem
    // clipGradientsKernel<<<numBlocks, blockSize>>>(grads, 5.0f, size);
    
    if (momentum_factor > 0.0f) {
        // Initialize momentum buffer if needed
        if (!momentum_buffer || buffer_size != size) {
            if (momentum_buffer) cudaFree(momentum_buffer);
            cudaMalloc(&momentum_buffer, size * sizeof(float));
            cudaMemset(momentum_buffer, 0, size * sizeof(float));
            buffer_size = size;
        }
        
        // Update with momentum
        updateWeightsWithMomentumKernel<<<numBlocks, blockSize>>>(
            params, grads, momentum_buffer, learning_rate, momentum_factor, size);
    } else {
        // Standard SGD
        updateWeightsKernel<<<numBlocks, blockSize>>>(params, grads, learning_rate, size);
    }
    
    cudaDeviceSynchronize();
}
