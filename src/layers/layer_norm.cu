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
    : d_model(d_model), epsilon(epsilon), gamma(1, d_model, 1.0f), beta(1, d_model, 0.0f),
      grad_gamma(1, d_model, 0.0f), grad_beta(1, d_model, 0.0f) {
    // Use a much larger epsilon for numerical stability and to prevent activation collapse
    if (epsilon < 1e-3) {
        this->epsilon = 1e-3;  // Much larger epsilon to prevent over-normalization
    }
}

Matrix LayerNorm::backward(const Matrix &grad_output, const Matrix &input) {
    int N = input.getRows();
    int D = input.getCols();
    Matrix grad_input(N, D, 0.0f);
    
    // CPU implementation for simplicity
    std::vector<float> h_input(N * D);
    std::vector<float> h_grad_output(N * D);
    std::vector<float> h_grad_input(N * D, 0.0f);
    std::vector<float> h_gamma(D);
    std::vector<float> h_grad_gamma(D, 0.0f);
    std::vector<float> h_grad_beta(D, 0.0f);
    
    input.copyToHost(h_input);
    grad_output.copyToHost(h_grad_output);
    gamma.copyToHost(h_gamma);
    
    for (int i = 0; i < N; i++) {
        // Calculate mean and variance
        float mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += h_input[i * D + j];
        }
        mean /= D;
        
        float variance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = h_input[i * D + j] - mean;
            variance += diff * diff;
        }
        variance /= D;
        
        float stddev = sqrtf(variance + epsilon);
        
        // Compute gradients
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        
        for (int j = 0; j < D; j++) {
            float xhat = (h_input[i * D + j] - mean) / stddev;
            float dy = h_grad_output[i * D + j];
            
            sum_dy += dy * h_gamma[j];
            sum_dy_xhat += dy * h_gamma[j] * xhat;
            
            // Accumulate parameter gradients
            h_grad_gamma[j] += dy * xhat;
            h_grad_beta[j] += dy;
        }
        
        // Compute input gradients
        for (int j = 0; j < D; j++) {
            float xhat = (h_input[i * D + j] - mean) / stddev;
            float dy = h_grad_output[i * D + j];
            
            h_grad_input[i * D + j] = (h_gamma[j] / stddev) * 
                (dy - sum_dy / D - xhat * sum_dy_xhat / D);
        }
    }
    
    grad_input.copyFromHost(h_grad_input);
    grad_gamma.copyFromHost(h_grad_gamma);
    grad_beta.copyFromHost(h_grad_beta);
    
    return grad_input;
}

void LayerNorm::updateWeights(float learning_rate) {
    // Update gamma and beta using accumulated gradients
    std::vector<float> h_gamma(d_model);
    std::vector<float> h_beta(d_model);
    std::vector<float> h_grad_gamma(d_model);
    std::vector<float> h_grad_beta(d_model);
    
    gamma.copyToHost(h_gamma);
    beta.copyToHost(h_beta);
    grad_gamma.copyToHost(h_grad_gamma);
    grad_beta.copyToHost(h_grad_beta);
    
    for (size_t i = 0; i < d_model; i++) {
        h_gamma[i] -= learning_rate * h_grad_gamma[i];
        h_beta[i] -= learning_rate * h_grad_beta[i];
        h_grad_gamma[i] = 0.0f; // Reset gradients
        h_grad_beta[i] = 0.0f;
    }
    
    gamma.copyFromHost(h_gamma);
    beta.copyFromHost(h_beta);
    grad_gamma.copyFromHost(h_grad_gamma);
    grad_beta.copyFromHost(h_grad_beta);
}