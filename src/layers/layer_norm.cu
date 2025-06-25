// src/layers/layer_norm.cu
#include "layer_norm.cuh"
#include <cuda_runtime.h>

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

        float stddev = sqrt(variance + epsilon);

        // Normalize and apply gamma and beta
        for (int j = 0; j < D; j++) {
            output[idx * D + j] = gamma[j] * (input[idx * D + j] - mean) / stddev + beta[j];
        }
    }
}

Matrix LayerNorm::forward(const Matrix &input) {
    int N = input.getRows();
    int D = input.getCols();
    Matrix output(N, D);

    float *d_input = input.getData();
    float *d_output = output.getData();
    float *d_gamma = gamma.getData();
    float *d_beta = beta.getData();

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    layer_norm_kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_gamma, d_beta, N, D, epsilon);

    cudaDeviceSynchronize();

    return output;
}

LayerNorm::LayerNorm(size_t d_model, double epsilon) 
    : d_model(d_model), epsilon(epsilon), gamma(1, d_model, 1.0f), beta(1, d_model, 0.0f) {
    // Constructor ya implementado con inicialización de gamma y beta
}