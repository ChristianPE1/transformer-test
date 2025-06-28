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

    // Use CPU implementation for stability
    std::vector<float> h_input(batch_size * input_dim);
    std::vector<float> h_weights(input_dim * output_dim);
    std::vector<float> h_bias(output_dim);
    std::vector<float> h_output(batch_size * output_dim, 0.0f);

    cudaMemcpy(h_input.data(), input.getData(), batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weights.data(), weights.getData(), input_dim * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bias.data(), bias.getData(), output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU matrix multiplication: output = input * weights + bias
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                sum += h_input[b * input_dim + i] * h_weights[i * output_dim + o];
            }
            h_output[b * output_dim + o] = sum + h_bias[o];
        }
    }

    Matrix output(batch_size, output_dim);
    cudaMemcpy(output.getData(), h_output.data(), batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    
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
    
    // Check for NaN/Inf gradients and clip them
    bool has_nan_weights = false, has_nan_bias = false;
    float max_grad_weight = 0.0f, max_grad_bias = 0.0f;
    
    for (size_t i = 0; i < h_grad_weights.size(); ++i) {
        if (std::isnan(h_grad_weights[i]) || std::isinf(h_grad_weights[i])) {
            h_grad_weights[i] = 0.0f;
            has_nan_weights = true;
        } else {
            // Clip gradients to prevent explosion
            h_grad_weights[i] = std::max(-1.0f, std::min(1.0f, h_grad_weights[i]));
            max_grad_weight = std::max(max_grad_weight, std::abs(h_grad_weights[i]));
        }
    }
    
    for (size_t i = 0; i < h_grad_bias.size(); ++i) {
        if (std::isnan(h_grad_bias[i]) || std::isinf(h_grad_bias[i])) {
            h_grad_bias[i] = 0.0f;
            has_nan_bias = true;
        } else {
            // Clip bias gradients
            h_grad_bias[i] = std::max(-1.0f, std::min(1.0f, h_grad_bias[i]));
            max_grad_bias = std::max(max_grad_bias, std::abs(h_grad_bias[i]));
        }
    }
    
    if (has_nan_weights || has_nan_bias) {
        std::cout << "[LINEAR] WARNING: Cleaned NaN/Inf gradients" << std::endl;
    }
    
    // std::cout << "[LINEAR] Max gradients - weights: " << max_grad_weight 
    //           << ", bias: " << max_grad_bias << std::endl;
    
    // Apply gradient descent with adaptive learning rate
    float effective_lr = learning_rate;
    
    // For output projection layers, use higher learning rate to break symmetry
    if (output_dim > 500) {
        effective_lr *= 2.0f; // Reducido de 5x a 2x para estabilidad
        std::cout << "[LINEAR] Output projection - using 2x learning rate: " << effective_lr << std::endl;
    }
    
    if (max_grad_weight > 0.5f || max_grad_bias > 0.5f) {
        effective_lr *= 0.5f; // Reduce learning rate if gradients are large
        std::cout << "[LINEAR] Large gradients detected, reducing lr to " << effective_lr << std::endl;
    }
    
    // Apply gradient descent: w = w - lr * grad_w
    for (size_t i = 0; i < h_weights.size(); ++i) {
        h_weights[i] -= effective_lr * h_grad_weights[i];
        
        // Less restrictive weight clipping - only prevent extreme values
        h_weights[i] = std::max(-5.0f, std::min(5.0f, h_weights[i]));
    }
    
    for (size_t i = 0; i < h_bias.size(); ++i) {
        h_bias[i] -= effective_lr * h_grad_bias[i];
        
        // Less restrictive bias clipping
        h_bias[i] = std::max(-3.0f, std::min(3.0f, h_bias[i]));
    }
    
    // Copy back to device
    weights.copyFromHost(h_weights);
    bias.copyFromHost(h_bias);
    
    // Clear stored gradients
    stored_grad_weights = Matrix(0, 0);
    stored_grad_bias = Matrix(0, 0);
    
    // std::cout << "[LINEAR] Weights updated" << std::endl;
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
    // Initialize weights with Xavier/Glorot initialization
    std::vector<float> weight_data(input_dim * output_dim);
    std::vector<float> bias_data(output_dim, 0.0f); // Initialize bias to zero
    
    // Use Xavier initialization
    float scale = sqrt(2.0f / (input_dim + output_dim)); // Xavier initialization
    
    // For output projection layers (large output_dim), use LARGER scale to increase variance
    if (output_dim > 500) {  // This is likely the output projection to vocab
        scale *= 3.0f;  // INCREASE scale for output layer to get more diverse predictions
        std::cout << "[LINEAR] Output projection layer detected, using LARGER scale for better variance" << std::endl;
    }
    
    // Reasonable scale range
    scale = std::max(0.01f, std::min(scale, 0.2f)); // Allow larger scales
    
    // Random initialization for weights
    srand(time(nullptr)); // Seed for reproducibility
    for (size_t i = 0; i < weight_data.size(); ++i) {
        // Generate random value in range [-scale, scale]
        float rand_val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        weight_data[i] = rand_val;
    }
    
    std::cout << "[LINEAR] Initialized " << input_dim << "x" << output_dim 
              << " with scale=" << scale << std::endl;
              
    weights.copyFromHost(weight_data);
    bias.copyFromHost(bias_data);
}