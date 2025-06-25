// filepath: /cuda-transformer/cuda-transformer/src/transformer/attention.cu
#include "attention.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>

#define MAX_SEQ_LEN 512

__device__ void softmax_device(float* data, int length) {
    float max_val = data[0];
    for (int i = 1; i < length; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < length; ++i) {
        data[i] /= sum;
    }
}

// Scaled Dot-Product Attention Kernel
__global__ void scaledDotProductAttentionKernel(
    const float* queries, const float* keys, const float* values, 
    float* output, const float* mask, 
    int seq_len, int d_k, int head_idx, int n_heads, int d_model) 
{
    int q_pos = blockIdx.x;  // Query position
    int k_pos = threadIdx.x; // Key position
    
    if (q_pos >= seq_len || k_pos >= seq_len) return;
    
    extern __shared__ float shared_mem[];
    float* attention_scores = shared_mem;
    float* values_cache = shared_mem + seq_len;
    
    // Calculate attention score: Q * K^T / sqrt(d_k)
    float score = 0.0f;
    int q_offset = q_pos * d_model + head_idx * d_k;
    int k_offset = k_pos * d_model + head_idx * d_k;
    
    for (int i = 0; i < d_k; ++i) {
        score += queries[q_offset + i] * keys[k_offset + i];
    }
    score /= sqrtf((float)d_k);
    
    // Apply mask if provided
    if (mask && mask[q_pos * seq_len + k_pos] == 0.0f) {
        score = -1e9f;
    }
    
    attention_scores[k_pos] = score;
    __syncthreads();
    
    // Apply softmax (only thread 0 per block)
    if (k_pos == 0) {
        softmax_device(attention_scores, seq_len);
    }
    __syncthreads();
    
    // Cache values for this head
    int v_offset = k_pos * d_model + head_idx * d_k;
    for (int i = 0; i < d_k; ++i) {
        values_cache[k_pos * d_k + i] = values[v_offset + i];
    }
    __syncthreads();
    
    // Compute weighted sum (parallel reduction)
    if (k_pos < d_k) {
        float weighted_sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            weighted_sum += attention_scores[i] * values_cache[i * d_k + k_pos];
        }
        
        int out_offset = q_pos * d_model + head_idx * d_k;
        output[out_offset + k_pos] = weighted_sum;
    }
}

// Multi-Head Attention combining all heads
__global__ void combineHeadsKernel(
    const float* multi_head_output, float* final_output,
    const float* W_O, int seq_len, int d_model, int n_heads) 
{
    int pos = blockIdx.x;
    int dim = threadIdx.x;
    
    if (pos >= seq_len || dim >= d_model) return;
    
    float sum = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        int head_dim = d_model / n_heads;
        for (int k = 0; k < head_dim; ++k) {
            int head_offset = pos * d_model + h * head_dim + k;
            int weight_offset = (h * head_dim + k) * d_model + dim;
            sum += multi_head_output[head_offset] * W_O[weight_offset];
        }
    }
    
    final_output[pos * d_model + dim] = sum;
}

Matrix MultiHeadAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask) {
    int seq_len = query.getRows();
    int d_model = query.getCols();
    int d_k = d_model / n_heads;

    // Create output matrices
    Matrix multi_head_output(seq_len, d_model);
    Matrix final_output(seq_len, d_model);
    
    // Initialize output to zero
    cudaMemset(multi_head_output.getData(), 0, seq_len * d_model * sizeof(float));
    cudaMemset(final_output.getData(), 0, seq_len * d_model * sizeof(float));

    // Process each head
    for (int h = 0; h < n_heads; ++h) {
        dim3 blockSize(seq_len);
        dim3 gridSize(seq_len);
        
        size_t shared_mem_size = (seq_len + seq_len * d_k) * sizeof(float);
        
        scaledDotProductAttentionKernel<<<gridSize, blockSize, shared_mem_size>>>(
            query.getData(),
            key.getData(), 
            value.getData(),
            multi_head_output.getData(),
            mask.getRows() > 0 ? mask.getData() : nullptr,
            seq_len,
            d_k,
            h,
            n_heads,
            d_model
        );
    }
    
    // Combine heads with output projection
    dim3 blockSize2(d_model);
    dim3 gridSize2(seq_len);
    
    combineHeadsKernel<<<gridSize2, blockSize2>>>(
        multi_head_output.getData(),
        final_output.getData(),
        W_O.getData(),
        seq_len,
        d_model,
        n_heads
    );
    
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    return final_output;
}

// Backward pass kernels
__global__ void attentionBackwardKernel(
    const float* grad_output, const float* attention_weights,
    const float* values, float* grad_values,
    int seq_len, int d_model, int head_idx, int n_heads) 
{
    int pos = blockIdx.x;
    int dim = threadIdx.x;
    int d_k = d_model / n_heads;
    
    if (pos >= seq_len || dim >= d_k) return;
    
    // Compute gradients w.r.t. values
    float grad_val = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        int att_idx = i * seq_len + pos;
        int grad_idx = i * d_model + head_idx * d_k + dim;
        grad_val += grad_output[grad_idx] * attention_weights[att_idx];
    }
    
    int val_idx = pos * d_model + head_idx * d_k + dim;
    grad_values[val_idx] = grad_val;
}

void MultiHeadAttention::backward(const Matrix &grad_output, Matrix &grad_query, Matrix &grad_key, Matrix &grad_value) {
    int seq_len = grad_output.getRows();
    int d_model = grad_output.getCols();
    int d_k = d_model / n_heads;
    
    // Initialize gradient matrices
    grad_query = Matrix(seq_len, d_model);
    grad_key = Matrix(seq_len, d_model);
    grad_value = Matrix(seq_len, d_model);
    
    cudaMemset(grad_query.getData(), 0, seq_len * d_model * sizeof(float));
    cudaMemset(grad_key.getData(), 0, seq_len * d_model * sizeof(float));
    cudaMemset(grad_value.getData(), 0, seq_len * d_model * sizeof(float));
    
    // Simple backward approximation - in a full implementation you'd store
    // attention weights from forward pass and compute proper gradients
    
    // For now, we'll use a simplified approach where gradients flow back
    // proportionally through the attention mechanism
    
    cudaMemcpy(grad_query.getData(), grad_output.getData(), seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(grad_key.getData(), grad_output.getData(), seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(grad_value.getData(), grad_output.getData(), seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
}

void MultiHeadAttention::updateWeights(float learning_rate) {
    // Update weight matrices using accumulated gradients
    // This is a simplified version - normally you'd accumulate gradients properly
    
    size_t weight_size = d_model * d_model;
    
    // Simple random perturbation as gradient update (placeholder)
    float* temp_updates = new float[weight_size];
    
    for (size_t i = 0; i < weight_size; ++i) {
        temp_updates[i] = ((float)rand() / RAND_MAX - 0.5f) * learning_rate * 0.1f;
    }
    
    // Apply updates (this should be proper gradient descent)
    float *d_temp;
    cudaMalloc(&d_temp, weight_size * sizeof(float));
    cudaMemcpy(d_temp, temp_updates, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Simple element-wise addition kernel (should be implemented properly)
    int threads = 256;
    int blocks = (weight_size + threads - 1) / threads;
    
    // For now, just add small random updates
    // In real implementation, this would be: W = W - lr * grad_W
    
    cudaFree(d_temp);
    delete[] temp_updates;
}

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads) 
    : d_model(d_model), n_heads(n_heads) {
    d_k = d_model / n_heads;
    d_v = d_model / n_heads;
    
    // Initialize weight matrices
    size_t weight_size = d_model * d_model;
    
    W_Q = Matrix(d_model, d_model);
    W_K = Matrix(d_model, d_model);
    W_V = Matrix(d_model, d_model);
    W_O = Matrix(d_model, d_model);
    
    // Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (d_model + d_model));
    
    // Initialize with random values (simplified)
    float* temp_data = new float[weight_size];
    for (size_t i = 0; i < weight_size; ++i) {
        temp_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    cudaMemcpy(W_Q.getData(), temp_data, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_K.getData(), temp_data, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_V.getData(), temp_data, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_O.getData(), temp_data, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] temp_data;
}

MultiHeadAttention::~MultiHeadAttention() {
    // Matrix destructor handles cleanup
}