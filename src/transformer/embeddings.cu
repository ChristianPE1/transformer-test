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

__global__ void initEmbeddingsKernel(float *embeddings, int vocab_size, int d_model, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size * d_model)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        embeddings[idx] = curand_normal(&state) * 0.1f; // Xavier-like initialization
    }
}

__global__ void embedLookupKernel(float *embeddings, int *input_ids, float *output,
                                  int vocab_size, int d_model, int seq_len)
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (token_idx < seq_len && dim_idx < d_model)
    {
        int token_id = input_ids[token_idx];
        if (token_id >= 0 && token_id < vocab_size)
        {
            // FIXED: Correct indexing for embedding lookup
            int embedding_idx = token_id * d_model + dim_idx;
            int output_idx = token_idx * d_model + dim_idx;
            output[output_idx] = embeddings[embedding_idx];
        }
        else
        {
            // Handle out-of-bounds tokens with zero
            int output_idx = token_idx * d_model + dim_idx;
            output[output_idx] = 0.0f;
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

    // Copy input tokens to device
    int *d_input_ids;
    cudaMalloc(&d_input_ids, seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, input_tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize output to zero first
    cudaMemset(output.getData(), 0, seq_len * d_model * sizeof(float));

    // Launch embedding lookup kernel with better grid configuration
    dim3 blockSize(16, 8);  // Reduced block size for better occupancy
    dim3 gridSize((seq_len + blockSize.x - 1) / blockSize.x,
                  (d_model + blockSize.y - 1) / blockSize.y);

    embedLookupKernel<<<gridSize, blockSize>>>(
        weights, d_input_ids, output.getData(), vocab_size, d_model, seq_len);

    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "[EMBEDDING] CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_input_ids);

    // DEBUG: Check if embedding weights are zero
    std::vector<float> sample_weights(10);
    cudaMemcpy(sample_weights.data(), weights, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[EMBEDDING] Sample weights: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(3) << sample_weights[i] << " ";
    }
    std::cout << std::endl;

    // DEBUG: Check if output is zero
    std::vector<float> sample_output(std::min(10, (int)(seq_len * d_model)));
    output.copyToHost(sample_output);
    std::cout << "[EMBEDDING] Sample output: ";
    for (int i = 0; i < std::min(5, (int)sample_output.size()); ++i) {
        std::cout << std::fixed << std::setprecision(3) << sample_output[i] << " ";
    }
    std::cout << std::endl;

    // DEBUG: Check input tokens
    std::cout << "[EMBEDDING] Input tokens: ";
    for (int i = 0; i < std::min(5, seq_len); ++i) {
        std::cout << input_tokens[i] << " ";
    }
    std::cout << std::endl;

    // DEBUG: Check specific embeddings for the input tokens
    if (!input_tokens.empty()) {
        std::vector<float> specific_embedding(d_model);
        int first_token = input_tokens[0];
        if (first_token >= 0 && first_token < (int)vocab_size) {
            cudaMemcpy(specific_embedding.data(), 
                      weights + first_token * d_model, 
                      d_model * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            std::cout << "[EMBEDDING] Token " << first_token << " embedding: ";
            for (int i = 0; i < std::min(5, (int)d_model); ++i) {
                std::cout << std::fixed << std::setprecision(3) << specific_embedding[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    return output;
}

// Add after existing Embedding code

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

__global__ void updateEmbeddingWeightsKernel(float* weights, float* gradients, 
                                            int* tokens, float learning_rate,
                                            int seq_len, int d_model, int vocab_size) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx < seq_len && dim_idx < d_model) {
        int token_id = tokens[token_idx];
        if (token_id >= 0 && token_id < vocab_size) {
            int weight_idx = token_id * d_model + dim_idx;
            int grad_idx = token_idx * d_model + dim_idx;
            
            // Aplicar gradiente con clipping para estabilidad
            float grad_value = gradients[grad_idx];
            
            // Gradient clipping
            if (grad_value > 1.0f) grad_value = 1.0f;
            if (grad_value < -1.0f) grad_value = -1.0f;
            
            // Actualización con momentum implícito
            float current_weight = weights[weight_idx];
            float update = learning_rate * grad_value;
            
            // Agregar algo de momentum para suavizar actualizaciones
            weights[weight_idx] = current_weight - update + 0.1f * update;
        }
    }
}

void Embedding::updateWeights(const Matrix& gradients, float learning_rate, const std::vector<int>& tokens) {
    int seq_len = tokens.size();
    
    // Copy tokens to device
    int *d_tokens;
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
    
    // Update weights
    dim3 blockSize(16, 16);
    dim3 gridSize((seq_len + blockSize.x - 1) / blockSize.x,
                  (d_model + blockSize.y - 1) / blockSize.y);
    
    updateEmbeddingWeightsKernel<<<gridSize, blockSize>>>(
        weights, gradients.getData(), d_tokens, learning_rate, seq_len, d_model, vocab_size);
    
    cudaDeviceSynchronize();
    cudaFree(d_tokens);
}