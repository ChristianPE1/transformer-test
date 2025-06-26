// filepath: cuda-transformer/cuda-transformer/src/layers/linear.cu
#include "linear.cuh"
#include "utils/cuda_utils.cuh"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>

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

// Backward pass for Linear layer
Matrix Linear::backward(const Matrix& grad_output, const Matrix& input) {
    int batch_size = grad_output.getRows();
    int output_dim = grad_output.getCols();
    
    // Compute gradients for weights and bias
    Matrix grad_weights(input_dim, output_dim, 0.0f);
    Matrix grad_bias(1, output_dim, 0.0f);
    Matrix grad_input(batch_size, input_dim, 0.0f);
    
    // Copy to host for CPU computation (simplified)
    std::vector<float> h_grad_output, h_input;
    grad_output.copyToHost(h_grad_output);
    input.copyToHost(h_input);
    
    std::vector<float> h_grad_weights(input_dim * output_dim, 0.0f);
    std::vector<float> h_grad_bias(output_dim, 0.0f);
    std::vector<float> h_grad_input(batch_size * input_dim, 0.0f);
    
    // Compute gradients
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_dim; ++i) {
            for (int o = 0; o < output_dim; ++o) {
                // Gradient w.r.t weights: grad_w = input^T * grad_output
                h_grad_weights[i * output_dim + o] += h_input[b * input_dim + i] * h_grad_output[b * output_dim + o];
            }
        }
        
        // Gradient w.r.t bias: grad_b = sum(grad_output)
        for (int o = 0; o < output_dim; ++o) {
            h_grad_bias[o] += h_grad_output[b * output_dim + o];
        }
    }
    
    // Compute gradient w.r.t input: grad_input = grad_output * W^T
    std::vector<float> h_weights;
    weights.copyToHost(h_weights);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_dim; ++i) {
            for (int o = 0; o < output_dim; ++o) {
                h_grad_input[b * input_dim + i] += h_grad_output[b * output_dim + o] * h_weights[i * output_dim + o];
            }
        }
    }
    
    // Store gradients for weight update
    grad_weights.copyFromHost(h_grad_weights);
    grad_bias.copyFromHost(h_grad_bias);
    grad_input.copyFromHost(h_grad_input);
    
    // Store gradients as member variables for later update
    stored_grad_weights = grad_weights;
    stored_grad_bias = grad_bias;
    
    return grad_input;
}

// Update weights using stored gradients
void Linear::updateWeights(float learning_rate) {
    if (stored_grad_weights.getRows() == 0) return; // No gradients stored
    
    // Copy current weights to host
    std::vector<float> h_weights, h_bias;
    std::vector<float> h_grad_weights, h_grad_bias;
    
    weights.copyToHost(h_weights);
    bias.copyToHost(h_bias);
    stored_grad_weights.copyToHost(h_grad_weights);
    stored_grad_bias.copyToHost(h_grad_bias);
    
    // Apply gradient descent: w = w - lr * grad_w
    for (size_t i = 0; i < h_weights.size(); ++i) {
        h_weights[i] -= learning_rate * h_grad_weights[i];
    }
    
    for (size_t i = 0; i < h_bias.size(); ++i) {
        h_bias[i] -= learning_rate * h_grad_bias[i];
    }
    
    // Copy back to device
    weights.copyFromHost(h_weights);
    bias.copyFromHost(h_bias);
    
    // Clear stored gradients
    stored_grad_weights = Matrix(0, 0);
    stored_grad_bias = Matrix(0, 0);
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
    // Initialize weights with He initialization (better for ReLU-like activations)
    std::vector<float> weight_data(input_dim * output_dim);
    std::vector<float> bias_data(output_dim, 0.0f); // Initialize bias to zero
    
    float scale = sqrt(2.0f / input_dim); // He initialization (better than Xavier for this case)
    
    // Random initialization for weights
    srand(time(nullptr)); // Seed for reproducibility
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    weights.copyFromHost(weight_data);
    bias.copyFromHost(bias_data);
}