// filepath: cuda-transformer/cuda-transformer/src/layers/linear.cu
#include "linear.cuh"
#include "utils/cuda_utils.cuh"
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ void linear_forward_kernel(const float *input, const float *weights, const float *bias, float *output, int input_dim, int output_dim, int batch_size) {
    int batch_index = blockIdx.x;
    int output_index = threadIdx.x;

    if (batch_index < batch_size && output_index < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += input[batch_index * input_dim + i] * weights[i * output_dim + output_index];
        }
        output[batch_index * output_dim + output_index] = sum + bias[output_index];
    }
}

Matrix Linear::forward(const Matrix &input) {
    int batch_size = input.getRows();
    int input_dim = input.getCols();
    int output_dim = weights.getCols();

    const float *d_input = input.getData();
    const float *d_weights = weights.getData();
    const float *d_bias = bias.getData();

    Matrix output(batch_size, output_dim);
    float *d_output = output.getData();

    linear_forward_kernel<<<batch_size, output_dim>>>(d_input, d_weights, d_bias, d_output, input_dim, output_dim, batch_size);

    cudaDeviceSynchronize();
    return output;
}

// Constructor
Linear::Linear(size_t input_dim, size_t output_dim) 
    : input_dim(input_dim), output_dim(output_dim), 
      weights(input_dim, output_dim), bias(1, output_dim) {
    initialize();
}

// Destructor
Linear::~Linear() {
    // Matrix destructor will handle cleanup automatically
}

// Initialize weights and bias
void Linear::initialize() {
    // Initialize weights with Xavier initialization
    std::vector<float> weight_data(input_dim * output_dim);
    std::vector<float> bias_data(output_dim, 0.0f); // Initialize bias to zero
    
    float scale = sqrt(2.0f / (input_dim + output_dim)); // Xavier initialization
    
    // Random initialization for weights
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    weights.copyFromHost(weight_data);
    bias.copyFromHost(bias_data);
}