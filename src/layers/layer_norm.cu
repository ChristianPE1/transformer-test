// src/layers/layer_norm.cu
#include "layer_norm.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

__global__ void layer_norm_kernel(float *input, float *output, float *gamma, float *beta, int N, int D, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mean = 0.0f;
        float variance = 0.0f;

        // Calculate mean
        for (int j = 0; j < D; j++) {
            mean += input[idx * D + j];
        }
        mean /= D;

        // Calculate variance
        for (int j = 0; j < D; j++) {
            float diff = input[idx * D + j] - mean;
            variance += diff * diff;
        }
        variance /= D;

        // CRITICAL FIX: Protect against division by zero and ensure epsilon is meaningful
        float stddev = sqrtf(variance + epsilon);
        
        // Additional protection: if stddev is still too small, use a minimum value
        if (stddev < 1e-3f) {
            stddev = 1e-3f;  // Larger minimum to prevent over-normalization
        }

        // Normalize and apply gamma and beta
        for (int j = 0; j < D; j++) {
            float normalized = (input[idx * D + j] - mean) / stddev;
            output[idx * D + j] = gamma[j] * normalized + beta[j];
        }
    }
}

Matrix LayerNorm::forward(const Matrix &input) {
    int N = input.getRows();
    int D = input.getCols();
    Matrix output(N, D);

    // Use CPU implementation for stability
    std::vector<float> h_input(N * D);
    std::vector<float> h_output(N * D);
    cudaMemcpy(h_input.data(), input.getData(), N * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Get gamma and beta values
    std::vector<float> h_gamma(D);
    std::vector<float> h_beta(D);
    cudaMemcpy(h_gamma.data(), gamma.getData(), D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta.data(), beta.getData(), D * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU LayerNorm implementation
    for (int i = 0; i < N; i++) {
        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += h_input[i * D + j];
        }
        mean /= D;

        // Calculate variance
        float variance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = h_input[i * D + j] - mean;
            variance += diff * diff;
        }
        variance /= D;

        // Calculate standard deviation with epsilon
        float stddev = sqrtf(variance + epsilon);

        // Normalize and apply gamma and beta
        for (int j = 0; j < D; j++) {
            float normalized = (h_input[i * D + j] - mean) / stddev;
            h_output[i * D + j] = h_gamma[j] * normalized + h_beta[j];
        }
    }

    // Copy result back to GPU
    cudaMemcpy(output.getData(), h_output.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

    return output;
}

LayerNorm::LayerNorm(size_t d_model, double epsilon) 
    : d_model(d_model), epsilon(epsilon), gamma(1, d_model, 1.0f), beta(1, d_model, 0.0f) {
    // Use a much larger epsilon for numerical stability and to prevent activation collapse
    if (epsilon < 1e-3) {
        this->epsilon = 1e-3;  // Much larger epsilon to prevent over-normalization
    }
}