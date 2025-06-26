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

// Corrected kernel for positional encoding following standard Transformer formula
__global__ void initPositionalEncodingKernel(float *pos_enc, int d_model, int max_len)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos < max_len && i < d_model)
    {
        // Correct positional encoding formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        
        if (i % 2 == 0)
        {
            // Even dimensions get sine: use i directly
            float div_term = powf(10000.0f, (float)i / (float)d_model);
            float angle = (float)pos / div_term;
            pos_enc[pos * d_model + i] = sinf(angle);
        }
        else
        {
            // Odd dimensions get cosine: use (i-1) to pair with previous even dimension
            float div_term = powf(10000.0f, (float)(i-1) / (float)d_model);
            float angle = (float)pos / div_term;
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

    std::cout << "[EMBEDDING] Forward pass - seq_len=" << seq_len << ", d_model=" << d_model << std::endl;
    
    // OPTIMIZED CPU-BASED APPROACH - More efficient than before
    // Only copy the embeddings we need instead of all weights
    std::vector<float> host_output(seq_len * d_model, 0.0f);
    
    // For each token, copy its embedding individually (more efficient)
    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_tokens[i];
        if (token_id >= 0 && token_id < (int)vocab_size) {
            // Copy just this token's embedding directly
            std::vector<float> token_embedding(d_model);
            cudaMemcpy(token_embedding.data(), 
                      weights + token_id * d_model, 
                      d_model * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            // Copy to output buffer
            for (int j = 0; j < (int)d_model; ++j) {
                host_output[i * d_model + j] = token_embedding[j];
            }
        } else {
            std::cout << "[EMBEDDING] WARNING: Invalid token_id " << token_id << " (vocab_size=" << vocab_size << ")" << std::endl;
            // Leave as zeros (already initialized)
        }
    }
    
    // Copy final result back to GPU
    cudaMemcpy(output.getData(), host_output.data(), seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // EXTENSIVE DEBUG OUTPUT
    std::cout << "[EMBEDDING] Input tokens: ";
    for (int i = 0; i < std::min(5, seq_len); ++i) {
        std::cout << input_tokens[i] << " ";
    }
    std::cout << std::endl;

    // Check for non-zero values in output
    int non_zero_count = 0;
    float sum = 0.0f, max_val = -1e10f, min_val = 1e10f;
    for (int i = 0; i < seq_len * d_model; ++i) {
        if (abs(host_output[i]) > 1e-6f) {
            non_zero_count++;
            sum += host_output[i];
            max_val = std::max(max_val, host_output[i]);
            min_val = std::min(min_val, host_output[i]);
        }
    }
    
    std::cout << "[EMBEDDING] Stats: " << non_zero_count << "/" << (seq_len * d_model) 
              << " non-zero values, sum=" << sum 
              << ", range=[" << min_val << ", " << max_val << "]" << std::endl;

    // Sample of actual values
    std::cout << "[EMBEDDING] First 10 output values: ";
    for (int i = 0; i < std::min(10, (int)host_output.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << host_output[i] << " ";
    }
    std::cout << std::endl;

    // Verify specific tokens
    if (!input_tokens.empty() && seq_len >= 1) {
        int first_token = input_tokens[0];
        if (first_token >= 0 && first_token < (int)vocab_size) {
            std::cout << "[EMBEDDING] Token " << first_token << " embedding values: ";
            for (int i = 0; i < std::min(8, (int)d_model); ++i) {
                std::cout << std::fixed << std::setprecision(6) << host_output[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    // CRITICAL: Make sure the Matrix output actually contains the right data
    // Let's verify the data was copied correctly to the device
    std::vector<float> verification(std::min(20, seq_len * (int)d_model));
    output.copyToHost(verification);
    std::cout << "[EMBEDDING] GPU verification - first 10 values: ";
    for (int i = 0; i < std::min(10, (int)verification.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << verification[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "[EMBEDDING] Forward pass completed - Output ready for encoder/decoder" << std::endl;
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

    // TEMPORARY FIX: Generate positional encoding on CPU to ensure it works
    std::vector<float> pos_enc_cpu(seq_len * d_model);
    
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < (int)d_model; i++) {
            if (i % 2 == 0) {
                // Even indices: sin(pos / 10000^(i/d_model))
                float div_term = pow(10000.0f, (float)i / (float)d_model);
                float angle = (float)pos / div_term;
                pos_enc_cpu[pos * d_model + i] = sin(angle);
            } else {
                // Odd indices: cos(pos / 10000^((i-1)/d_model))
                float div_term = pow(10000.0f, (float)(i-1) / (float)d_model);
                float angle = (float)pos / div_term;
                pos_enc_cpu[pos * d_model + i] = cos(angle);
            }
        }
    }
    
    // Copy CPU-generated positional encoding to GPU
    cudaMemcpy(result.getData(), pos_enc_cpu.data(), 
               seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);

    // DEBUG: Verify positional encoding values
    float pos_sum = 0.0f;
    int non_zero_pos = 0;
    float pos_max = -1e10f, pos_min = 1e10f;
    
    for (int i = 0; i < std::min(20, seq_len * (int)d_model); ++i) {
        if (abs(pos_enc_cpu[i]) > 1e-8f) {
            non_zero_pos++;
            pos_sum += pos_enc_cpu[i];
            pos_max = std::max(pos_max, pos_enc_cpu[i]);
            pos_min = std::min(pos_min, pos_enc_cpu[i]);
        }
    }
    
    std::cout << "[POS_ENC] Stats: " << non_zero_pos << "/" << std::min(20, seq_len * (int)d_model)
              << " non-zero, sum=" << pos_sum 
              << ", range=[" << pos_min << ", " << pos_max << "]" << std::endl;
    
    std::cout << "[POS_ENC] First 10 values: ";
    for (int i = 0; i < std::min(10, seq_len * (int)d_model); ++i) {
        std::cout << std::fixed << std::setprecision(6) << pos_enc_cpu[i] << " ";
    }
    std::cout << std::endl;

    return result;
}

void Embedding::updateWeights(const Matrix& gradients, float learning_rate, const std::vector<int>& tokens) {
    int seq_len = tokens.size();
    
    std::cout << "[EMBEDDING] Updating weights for " << seq_len << " tokens with lr=" << learning_rate << std::endl;
    
    // Copy gradients to host for processing
    std::vector<float> grad_data(seq_len * d_model);
    gradients.copyToHost(grad_data);
    
    // Compute gradient statistics for debugging
    float grad_sum = 0.0f, grad_max = -1e10f, grad_min = 1e10f;
    int non_zero_grads = 0;
    for (int i = 0; i < seq_len * d_model; ++i) {
        float g = grad_data[i];
        if (abs(g) > 1e-8f) {
            non_zero_grads++;
            grad_sum += g;
            grad_max = std::max(grad_max, g);
            grad_min = std::min(grad_min, g);
        }
    }
    
    std::cout << "[EMBEDDING] Gradient stats: " << non_zero_grads << "/" << (seq_len * d_model) 
              << " non-zero, sum=" << grad_sum
              << ", range=[" << grad_min << ", " << grad_max << "]" << std::endl;
    
    if (non_zero_grads == 0) {
        std::cout << "[EMBEDDING] WARNING: All gradients are zero - no update will occur!" << std::endl;
        return;
    }
    
    // Update weights token by token (more efficient than before)
    for (int i = 0; i < seq_len; ++i) {
        int token_id = tokens[i];
        if (token_id >= 0 && token_id < (int)vocab_size) {
            // Copy current embedding to host
            std::vector<float> current_embedding(d_model);
            cudaMemcpy(current_embedding.data(), 
                      weights + token_id * d_model, 
                      d_model * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            // Apply gradients with clipping
            bool has_update = false;
            for (int j = 0; j < (int)d_model; ++j) {
                float grad = grad_data[i * d_model + j];
                
                // Gradient clipping for stability
                if (grad > 5.0f) grad = 5.0f;
                if (grad < -5.0f) grad = -5.0f;
                
                if (abs(grad) > 1e-8f) {
                    current_embedding[j] -= learning_rate * grad;
                    has_update = true;
                }
            }
            
            // Copy back to device only if there was an actual update
            if (has_update) {
                cudaMemcpy(weights + token_id * d_model,
                          current_embedding.data(),
                          d_model * sizeof(float),
                          cudaMemcpyHostToDevice);
            }
        }
    }
    
    cudaDeviceSynchronize();
    
    // Verify that weights actually changed
    std::vector<float> sample_weights_after(10);
    cudaMemcpy(sample_weights_after.data(), weights, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[EMBEDDING] Sample weights after update: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(8) << sample_weights_after[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "[EMBEDDING] Weight update completed successfully" << std::endl;
}