#include "embeddings.cuh"
#include "../utils/matrix.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>

// Simple kernel for positional encoding (keeping only this one for now)
__global__ void initPositionalEncodingKernel(float *pos_enc, int d_model, int max_len)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos < max_len && i < d_model)
    {
        float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / d_model);
        if (i % 2 == 0)
        {
            pos_enc[pos * d_model + i] = sinf(angle);
        }
        else
        {
            pos_enc[pos * d_model + i] = cosf(angle);
        }
    }
}

Embedding::Embedding(size_t vocab_size, size_t d_model)
    : vocab_size(vocab_size), d_model(d_model)
{

    cudaMalloc(&weights, vocab_size * d_model * sizeof(float));
    initializeWeights();
}

Embedding::~Embedding()
{
    if (weights)
    {
        cudaFree(weights);
    }
}

void Embedding::initializeWeights()
{
    // SIMPLIFIED AND ROBUST INITIALIZATION
    int total_size = vocab_size * d_model;
    
    // Initialize on host first
    std::vector<float> host_weights(total_size);
    
    // Simple random initialization on host
    srand(time(nullptr));
    for (int i = 0; i < total_size; ++i) {
        // Xavier initialization: uniform distribution in [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
        float range = sqrt(6.0f / (vocab_size + d_model));
        host_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * range;
        
        // Ensure non-zero values for debugging
        if (abs(host_weights[i]) < 1e-6f) {
            host_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
    
    // Copy to device
    cudaMemcpy(weights, host_weights.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    // DEBUG: Verify initialization worked
    std::vector<float> verify_weights(10);
    cudaMemcpy(verify_weights.data(), weights, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[EMBEDDING] INIT - Sample weights after initialization: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(6) << verify_weights[i] << " ";
    }
    std::cout << std::endl;
}

Matrix Embedding::forward(const std::vector<int> &input_tokens)
{
    int seq_len = input_tokens.size();
    Matrix output(seq_len, d_model);

    std::cout << "[EMBEDDING] Using pure CUDA memory approach (no kernels)" << std::endl;
    
    // DEBUG: Verify weights are initialized
    std::vector<float> sample_weights(10);
    cudaMemcpy(sample_weights.data(), weights, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[EMBEDDING] Sample weights BEFORE lookup: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(6) << sample_weights[i] << " ";
    }
    std::cout << std::endl;
    
    // Check CUDA error after weight check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[EMBEDDING] CUDA Error after weight check: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Initialize output to zero using standard CUDA
    cudaMemset(output.getData(), 0, seq_len * d_model * sizeof(float));
    
    // Copy embeddings row by row using only cudaMemcpy (most compatible)
    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_tokens[i];
        if (token_id >= 0 && token_id < (int)vocab_size) {
            // Calculate source and destination pointers
            float* src = weights + token_id * d_model;
            float* dst = output.getData() + i * d_model;
            
            // Use the most basic CUDA memory copy
            cudaError_t copy_result = cudaMemcpy(dst, src, d_model * sizeof(float), cudaMemcpyDeviceToDevice);
            
            if (copy_result != cudaSuccess) {
                std::cout << "[EMBEDDING] Copy error for token " << token_id << ": " << cudaGetErrorString(copy_result) << std::endl;
            }
        } else {
            std::cout << "[EMBEDDING] WARNING: Invalid token_id " << token_id << " (vocab_size=" << vocab_size << ")" << std::endl;
        }
    }
    
    // Ensure all copies are complete
    cudaDeviceSynchronize();
    
    // Check for any CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[EMBEDDING] CUDA Error after copies: " << cudaGetErrorString(err) << std::endl;
    }

    // DEBUG: Check output after lookup
    std::vector<float> sample_output(std::min(20, (int)(seq_len * d_model)));
    output.copyToHost(sample_output);
    std::cout << "[EMBEDDING] Sample output AFTER lookup: ";
    for (int i = 0; i < std::min(10, (int)sample_output.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << sample_output[i] << " ";
    }
    std::cout << std::endl;

    // DEBUG: Input tokens
    std::cout << "[EMBEDDING] Input tokens: ";
    for (int i = 0; i < std::min(5, seq_len); ++i) {
        std::cout << input_tokens[i] << " ";
    }
    std::cout << std::endl;

    // Verify specific embedding was copied correctly
    if (!input_tokens.empty()) {
        int first_token = input_tokens[0];
        if (first_token >= 0 && first_token < (int)vocab_size) {
            std::vector<float> original_embedding(10);
            std::vector<float> copied_embedding(10);
            
            // Get original from weights
            cudaMemcpy(original_embedding.data(), 
                      weights + first_token * d_model, 
                      10 * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            // Get copied from output
            cudaMemcpy(copied_embedding.data(), 
                      output.getData(), 
                      10 * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            std::cout << "[EMBEDDING] Token " << first_token << " original: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << std::fixed << std::setprecision(6) << original_embedding[i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "[EMBEDDING] Token " << first_token << " copied: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << std::fixed << std::setprecision(6) << copied_embedding[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    return output;
}

PositionalEncoding::PositionalEncoding(size_t d_model, size_t max_len)
    : d_model(d_model), max_len(max_len)
{

    cudaMalloc(&pos_encodings, max_len * d_model * sizeof(float));
    initializeEncodings();
}

PositionalEncoding::~PositionalEncoding()
{
    if (pos_encodings)
    {
        cudaFree(pos_encodings);
    }
}

void PositionalEncoding::initializeEncodings()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((max_len + blockSize.x - 1) / blockSize.x,
                  (d_model + blockSize.y - 1) / blockSize.y);

    initPositionalEncodingKernel<<<gridSize, blockSize>>>(pos_encodings, d_model, max_len);
    cudaDeviceSynchronize();
}

Matrix PositionalEncoding::getEncoding(int seq_len)
{
    Matrix result(seq_len, d_model);

    // Copy relevant portion of positional encodings
    cudaMemcpy(result.getData(), pos_encodings,
               seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

    return result;
}

void Embedding::updateWeights(const Matrix& gradients, float learning_rate, const std::vector<int>& tokens) {
    int seq_len = tokens.size();
    
    std::cout << "[EMBEDDING] Updating weights for " << seq_len << " tokens with lr=" << learning_rate << std::endl;
    
    // Simple host-based weight update to ensure compatibility
    std::vector<float> grad_data(seq_len * d_model);
    gradients.copyToHost(grad_data);
    
    // Update weights token by token
    for (int i = 0; i < seq_len; ++i) {
        int token_id = tokens[i];
        if (token_id >= 0 && token_id < (int)vocab_size) {
            // Copy current embedding to host
            std::vector<float> current_embedding(d_model);
            cudaMemcpy(current_embedding.data(), 
                      weights + token_id * d_model, 
                      d_model * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            // Apply gradients
            for (int j = 0; j < (int)d_model; ++j) {
                float grad = grad_data[i * d_model + j];
                // Gradient clipping
                if (grad > 1.0f) grad = 1.0f;
                if (grad < -1.0f) grad = -1.0f;
                
                current_embedding[j] -= learning_rate * grad;
            }
            
            // Copy back to device
            cudaMemcpy(weights + token_id * d_model,
                      current_embedding.data(),
                      d_model * sizeof(float),
                      cudaMemcpyHostToDevice);
        }
    }
    
    cudaDeviceSynchronize();
    std::cout << "[EMBEDDING] Weight update completed" << std::endl;
}