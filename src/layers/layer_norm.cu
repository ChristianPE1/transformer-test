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
        if (stddev < 1e-8f) {
            stddev = 1e-8f;
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

    // DEBUG: Check input values
    std::vector<float> debug_input(std::min(10, N * D));
    cudaMemcpy(debug_input.data(), input.getData(), debug_input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    int non_zero_input = 0;
    for (int i = 0; i < debug_input.size(); i++) {
        if (abs(debug_input[i]) > 1e-8f) non_zero_input++;
    }
    
    printf("[LAYER_NORM] Input: %d/%d non-zero, first 5: ", non_zero_input, (int)debug_input.size());
    for (int i = 0; i < std::min(5, (int)debug_input.size()); i++) {
        printf("%.6f ", debug_input[i]);
    }
    printf("\n");

    float *d_input = input.getData();
    float *d_output = output.getData();
    float *d_gamma = gamma.getData();
    float *d_beta = beta.getData();

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    layer_norm_kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_gamma, d_beta, N, D, epsilon);

    cudaDeviceSynchronize();

    // DEBUG: Check output values
    std::vector<float> debug_output(std::min(10, N * D));
    cudaMemcpy(debug_output.data(), output.getData(), debug_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    int non_zero_output = 0;
    for (int i = 0; i < debug_output.size(); i++) {
        if (abs(debug_output[i]) > 1e-8f) non_zero_output++;
    }
    
    printf("[LAYER_NORM] Output: %d/%d non-zero, first 5: ", non_zero_output, (int)debug_output.size());
    for (int i = 0; i < std::min(5, (int)debug_output.size()); i++) {
        printf("%.6f ", debug_output[i]);
    }
    printf("\n");

    return output;
}

LayerNorm::LayerNorm(size_t d_model, double epsilon) 
    : d_model(d_model), epsilon(epsilon), gamma(1, d_model, 1.0f), beta(1, d_model, 0.0f) {
    // Use a larger epsilon for numerical stability
    if (epsilon < 1e-5) {
        this->epsilon = 1e-5;  // Minimum epsilon for stability
    }
}