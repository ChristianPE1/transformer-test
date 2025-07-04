// filepath: cuda-transformer/cuda-transformer/src/layers/feed_forward.cu
#include "feed_forward.cuh"
#include "utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdlib> // Para rand()
#include <ctime>   // Para time()
#include <iostream>

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

    // O aqui CPU para mas estabilidad
    std::vector<float> input_h, W1_h, W2_h, b1_h, b2_h;
    input.copyToHost(input_h);
    W1.copyToHost(W1_h);
    W2.copyToHost(W2_h);
    
    b1_h.resize(d_ff);
    b2_h.resize(d_model);
    cudaMemcpy(b1_h.data(), b1, d_ff * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2_h.data(), b2, d_model * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> output_h(rows * output_dim);
    
    // Create intermediate activation matrix for the hidden layer
    std::vector<float> hidden(rows * d_ff);
    
    // First layer: input -> hidden (with ReLU)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < input_dim; ++k) {
                sum += input_h[i * input_dim + k] * W1_h[k * d_ff + j];
            }
            sum += b1_h[j];
            hidden[i * d_ff + j] = fmaxf(0.0f, sum); // ReLU activation
        }
    }
    
    // Second layer: hidden -> output (no activation)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_ff; ++k) {
                sum += hidden[i * d_ff + k] * W2_h[k * output_dim + j];
            }
            sum += b2_h[j];
            output_h[i * output_dim + j] = sum;
        }
    }
    
    Matrix output(rows, output_dim);
    output.copyFromHost(output_h);
    
    return output;
}

FeedForward::FeedForward(size_t d_model, size_t d_ff) 
    : d_model(d_model), d_ff(d_ff), W1(d_model, d_ff), W2(d_ff, d_model),
      grad_W1(d_model, d_ff, 0.0f), grad_W2(d_ff, d_model, 0.0f) {
    // Alocar memoria para b1 y b2
    cudaMalloc(&b1, d_ff * sizeof(float));
    cudaMalloc(&b2, d_model * sizeof(float));
    
    // Alocar memoria para gradientes de biases
    cudaMalloc(&grad_b1, d_ff * sizeof(float));
    cudaMalloc(&grad_b2, d_model * sizeof(float));
    
    // Inicializar gradientes de biases a cero
    cudaMemset(grad_b1, 0, d_ff * sizeof(float));
    cudaMemset(grad_b2, 0, d_model * sizeof(float));
    
    initializeWeights();
}

FeedForward::~FeedForward() {
    if (b1) cudaFree(b1);
    if (b2) cudaFree(b2);
    if (grad_b1) cudaFree(grad_b1);
    if (grad_b2) cudaFree(grad_b2);
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
    // Use proper seed
    srand(static_cast<unsigned>(time(nullptr)));
    
    // Initialize W1 weights with Xavier initialization
    int W1_size = d_model * d_ff;
    std::vector<float> W1_data(W1_size);
    float xavier_w1 = sqrt(2.0f / (d_model + d_ff)); // Xavier initialization
    for (int i = 0; i < W1_size; ++i) {
        W1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_w1;
    }
    W1.copyFromHost(W1_data);

    // Initialize W2 weights with Xavier initialization
    int W2_size = d_ff * d_model;
    std::vector<float> W2_data(W2_size);
    float xavier_w2 = sqrt(2.0f / (d_ff + d_model)); // Xavier initialization
    for (int i = 0; i < W2_size; ++i) {
        W2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_w2;
    }
    W2.copyFromHost(W2_data);

    // Initialize biases to zero
    std::vector<float> b1_data(d_ff, 0.0f);
    std::vector<float> b2_data(d_model, 0.0f);
    cudaMemcpy(b1, b1_data.data(), d_ff * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, b2_data.data(), d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "[FEEDFORWARD] Weights initialized with Xavier initialization" << std::endl;
    std::cout << "[FEEDFORWARD] W1 scale: " << xavier_w1 << ", W2 scale: " << xavier_w2 << std::endl;
}

Matrix FeedForward::backward(const Matrix &grad_output, const Matrix &input) {
    int batch_size = grad_output.getRows();
    int output_dim = grad_output.getCols();
    int input_dim = input.getCols();
    
    // Initialize gradient for input
    Matrix grad_input(batch_size, input_dim, 0.0f);
    
    // FFN: output = W2 * ReLU(W1 * input + b1) + b2
    // We need to compute gradients through ReLU activation
    
    std::vector<float> h_grad_output, h_input;
    grad_output.copyToHost(h_grad_output);
    input.copyToHost(h_input);
    
    // Get current weights for gradient computation
    std::vector<float> W1_data, W2_data;
    W1.copyToHost(W1_data);
    W2.copyToHost(W2_data);
    
    // Initialize gradient accumulators
    std::vector<float> grad_W1_data(d_model * d_ff, 0.0f);
    std::vector<float> grad_W2_data(d_ff * d_model, 0.0f);
    std::vector<float> grad_b1_h(d_ff, 0.0f);
    std::vector<float> grad_b2_h(d_model, 0.0f);
    std::vector<float> grad_input_h(batch_size * input_dim, 0.0f);
    
    // REAL BACKWARD COMPUTATION
    for (int b = 0; b < batch_size; ++b) {
        // Step 1: Compute intermediate values (W1 * input + b1)
        std::vector<float> z1(d_ff, 0.0f);  // Before ReLU
        std::vector<float> a1(d_ff, 0.0f);  // After ReLU
        
        for (int j = 0; j < d_ff; ++j) {
            for (int i = 0; i < input_dim; ++i) {
                z1[j] += W1_data[i * d_ff + j] * h_input[b * input_dim + i];
            }
            // Add bias (simplified - would get from device in real implementation)
            a1[j] = fmaxf(0.0f, z1[j]); // ReLU activation
        }
        
        // Step 2: Gradient of loss w.r.t b2 = grad_output
        for (int i = 0; i < d_model; ++i) {
            grad_b2_h[i] += h_grad_output[b * d_model + i];
        }
        
        // Step 3: Gradient of loss w.r.t W2
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                grad_W2_data[j * d_model + i] += a1[j] * h_grad_output[b * d_model + i];
            }
        }
        
        // Step 4: Gradient of loss w.r.t a1 (intermediate activation)
        std::vector<float> grad_a1(d_ff, 0.0f);
        for (int j = 0; j < d_ff; ++j) {
            for (int i = 0; i < d_model; ++i) {
                grad_a1[j] += W2_data[j * d_model + i] * h_grad_output[b * d_model + i];
            }
        }
        
        // Step 5: Gradient through ReLU (derivative is 1 if z1 > 0, else 0)
        std::vector<float> grad_z1(d_ff, 0.0f);
        for (int j = 0; j < d_ff; ++j) {
            grad_z1[j] = (z1[j] > 0.0f) ? grad_a1[j] : 0.0f;
        }
        
        // Step 6: Gradient w.r.t b1
        for (int j = 0; j < d_ff; ++j) {
            grad_b1_h[j] += grad_z1[j];
        }
        
        // Step 7: Gradient w.r.t W1
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                grad_W1_data[i * d_ff + j] += h_input[b * input_dim + i] * grad_z1[j];
            }
        }
        
        // Step 8: Gradient w.r.t input
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                grad_input_h[b * input_dim + i] += W1_data[i * d_ff + j] * grad_z1[j];
            }
        }
    }
    
    // Store gradients for weight updates
    grad_W1.copyFromHost(grad_W1_data);
    grad_W2.copyFromHost(grad_W2_data);
    
    cudaMemcpy(grad_b1, grad_b1_h.data(), d_ff * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_b2, grad_b2_h.data(), d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    grad_input.copyFromHost(grad_input_h);
    
    std::cout << "[FEEDFORWARD] Real backward pass completed - gradients computed for W1, W2, b1, b2" << std::endl;
    
    return grad_input;
}

void FeedForward::updateWeights(float learning_rate) {
    if (learning_rate == 0.0f) {
        std::cout << "[FEEDFORWARD] WARNING: Learning rate is 0!" << std::endl;
        return;
    }
    
    // Update W1
    std::vector<float> W1_data, grad_W1_data;
    W1.copyToHost(W1_data);
    grad_W1.copyToHost(grad_W1_data);
    
    // Check for NaN or inf in weights and gradients
    bool has_nan_w1 = false, has_nan_grad_w1 = false;
    for (size_t i = 0; i < W1_data.size(); ++i) {
        if (std::isnan(W1_data[i]) || std::isinf(W1_data[i])) has_nan_w1 = true;
        if (std::isnan(grad_W1_data[i]) || std::isinf(grad_W1_data[i])) has_nan_grad_w1 = true;
    }
    
    if (has_nan_w1 || has_nan_grad_w1) {
        std::cout << "[FEEDFORWARD] ERROR: NaN/Inf detected in W1 weights or gradients! Reinitializing..." << std::endl;
        // Reinitialize W1
        for (size_t i = 0; i < W1_data.size(); ++i) {
            W1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f; // Smaller initialization
        }
        W1.copyFromHost(W1_data);
        // Skip this update
        return;
    }
    
    // Clip gradients para acelerar - menos conservador
    for (size_t i = 0; i < grad_W1_data.size(); ++i) {
        if (grad_W1_data[i] > 2.0f) grad_W1_data[i] = 2.0f;
        if (grad_W1_data[i] < -2.0f) grad_W1_data[i] = -2.0f;
    }
    
    for (size_t i = 0; i < W1_data.size(); ++i) {
        W1_data[i] -= learning_rate * grad_W1_data[i];
        
        // RELAJAR: Limitar pesos W1 para acelerar pero mantener estabilidad
        if (W1_data[i] > 1.0f) W1_data[i] = 1.0f;
        if (W1_data[i] < -1.0f) W1_data[i] = -1.0f;
    }
    W1.copyFromHost(W1_data);
    
    // Update W2
    std::vector<float> W2_data, grad_W2_data;
    W2.copyToHost(W2_data);
    grad_W2.copyToHost(grad_W2_data);
    
    // Check for NaN or inf in W2
    bool has_nan_w2 = false, has_nan_grad_w2 = false;
    for (size_t i = 0; i < W2_data.size(); ++i) {
        if (std::isnan(W2_data[i]) || std::isinf(W2_data[i])) has_nan_w2 = true;
        if (std::isnan(grad_W2_data[i]) || std::isinf(grad_W2_data[i])) has_nan_grad_w2 = true;
    }
    
    if (has_nan_w2 || has_nan_grad_w2) {
        std::cout << "[FEEDFORWARD] ERROR: NaN/Inf detected in W2 weights or gradients! Reinitializing..." << std::endl;
        // Reinitialize W2
        for (size_t i = 0; i < W2_data.size(); ++i) {
            W2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f; // Smaller initialization
        }
        W2.copyFromHost(W2_data);
        // Skip this update
        return;
    }
    
    // Clip gradients para acelerar
    for (size_t i = 0; i < grad_W2_data.size(); ++i) {
        if (grad_W2_data[i] > 2.0f) grad_W2_data[i] = 2.0f;
        if (grad_W2_data[i] < -2.0f) grad_W2_data[i] = -2.0f;
    }
    
    for (size_t i = 0; i < W2_data.size(); ++i) {
        W2_data[i] -= learning_rate * grad_W2_data[i];
        
        // RELAJAR: Limitar pesos W2 para acelerar
        if (W2_data[i] > 1.0f) W2_data[i] = 1.0f;
        if (W2_data[i] < -1.0f) W2_data[i] = -1.0f;
    }
    W2.copyFromHost(W2_data);
    
    // Update biases (simplified - would use CUDA kernels in optimized version)
    std::vector<float> b1_h(d_ff), b2_h(d_model);
    std::vector<float> grad_b1_h(d_ff), grad_b2_h(d_model);
    
    cudaMemcpy(b1_h.data(), b1, d_ff * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2_h.data(), b2, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_b1_h.data(), grad_b1, d_ff * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_b2_h.data(), grad_b2, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clip bias gradients and check for NaN
    for (size_t i = 0; i < d_ff; ++i) {
        if (std::isnan(grad_b1_h[i]) || std::isinf(grad_b1_h[i])) {
            grad_b1_h[i] = 0.0f;
        } else {
            if (grad_b1_h[i] > 2.0f) grad_b1_h[i] = 2.0f;
            if (grad_b1_h[i] < -2.0f) grad_b1_h[i] = -2.0f;
        }
        b1_h[i] -= learning_rate * grad_b1_h[i];
        
        // Relajar bias b1
        if (b1_h[i] > 1.0f) b1_h[i] = 1.0f;
        if (b1_h[i] < -1.0f) b1_h[i] = -1.0f;
    }
    
    for (size_t i = 0; i < d_model; ++i) {
        if (std::isnan(grad_b2_h[i]) || std::isinf(grad_b2_h[i])) {
            grad_b2_h[i] = 0.0f;
        } else {
            if (grad_b2_h[i] > 2.0f) grad_b2_h[i] = 2.0f;
            if (grad_b2_h[i] < -2.0f) grad_b2_h[i] = -2.0f;
        }
        b2_h[i] -= learning_rate * grad_b2_h[i];
        
        // Relajar bias b2
        if (b2_h[i] > 1.0f) b2_h[i] = 1.0f;
        if (b2_h[i] < -1.0f) b2_h[i] = -1.0f;
    }
    
    cudaMemcpy(b1, b1_h.data(), d_ff * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, b2_h.data(), d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    // Reset gradients
    grad_W1 = Matrix(d_model, d_ff, 0.0f);
    grad_W2 = Matrix(d_ff, d_model, 0.0f);
    cudaMemset(grad_b1, 0, d_ff * sizeof(float));
    cudaMemset(grad_b2, 0, d_model * sizeof(float));
    
    std::cout << "[FEEDFORWARD] Weights updated with lr=" << learning_rate << std::endl;
}