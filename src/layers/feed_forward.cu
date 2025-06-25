// filepath: cuda-transformer/cuda-transformer/src/layers/feed_forward.cu
#include "feed_forward.cuh"
#include "utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdlib> // Para rand()

__global__ void feedForwardKernel(
    const float* input, float* output,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    int rows, int input_dim, int d_ff, int output_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        // First layer: Linear + ReLU
        for (int j = 0; j < d_ff; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < input_dim; ++k) {
                sum += input[idx * input_dim + k] * W1[k * d_ff + j];
            }
            sum += b1[j];
            output[idx * d_ff + j] = fmaxf(0.0f, sum); // ReLU
        }

        // Second layer: Linear
        for (int j = 0; j < output_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_ff; ++k) {
                sum += output[idx * d_ff + k] * W2[k * output_dim + j];
            }
            sum += b2[j];
            output[idx * output_dim + j] = sum; // No activation
        }
    }
}

Matrix FeedForward::forward(const Matrix &input) {
    int rows = input.getRows();
    int input_dim = input.getCols();
    int d_ff = this->d_ff;
    int output_dim = this->d_model;

    Matrix output(rows, output_dim);

    // Asume que W1, W2, b1, b2 están en memoria de dispositivo
    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    feedForwardKernel<<<numBlocks, blockSize>>>(
        input.getData(), output.getData(),
        W1.getData(), b1,
        W2.getData(), b2,
        rows, input_dim, d_ff, output_dim
    );
    cudaDeviceSynchronize();

    return output;
}

FeedForward::FeedForward(size_t d_model, size_t d_ff) 
    : d_model(d_model), d_ff(d_ff), W1(d_model, d_ff), W2(d_ff, d_model) {
    // Alocar memoria para b1 y b2
    cudaMalloc(&b1, d_ff * sizeof(float));
    cudaMalloc(&b2, d_model * sizeof(float));
    initializeWeights();
}

FeedForward::~FeedForward() {
    if (b1) cudaFree(b1);
    if (b2) cudaFree(b2);
}

__global__ void initializeWeightsKernel(float* weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state) * 0.1f; // Xavier-like initialization
    }
}

void FeedForward::initializeWeights() {
    // Initialize W1 weights
    int W1_size = d_model * d_ff;
    std::vector<float> W1_data(W1_size);
    for (int i = 0; i < W1_size; ++i) {
        W1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // Random initialization
    }
    W1.copyFromHost(W1_data);

    // Initialize W2 weights
    int W2_size = d_ff * d_model;
    std::vector<float> W2_data(W2_size);
    for (int i = 0; i < W2_size; ++i) {
        W2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // Random initialization
    }
    W2.copyFromHost(W2_data);

    // Initialize biases to zero (ya están inicializados con cudaMemset en el constructor)
}