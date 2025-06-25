// filepath: /cuda-transformer/cuda-transformer/src/transformer/attention.cu
#include "attention.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>

#define MAX_SEQ_LEN 256

__device__ void softmax(float* data, int length) {
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

__global__ void multiHeadAttentionKernel(
    const float* queries, const float* keys, const float* values, 
    float* output, int d_model, int n_heads, int seq_length) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_size = d_model / n_heads;

    if (idx < seq_length && seq_length <= MAX_SEQ_LEN) {
        float attention_scores[MAX_SEQ_LEN];

        // Calcula los scores de atención
        for (int i = 0; i < seq_length; ++i) {
            float score = 0.0f;
            for (int j = 0; j < head_size; ++j) {
                score += queries[idx * d_model + j] * keys[i * d_model + j];
            }
            attention_scores[i] = score;
        }

        // Softmax sobre los scores
        softmax(attention_scores, seq_length);

        // Calcula la salida ponderada
        for (int i = 0; i < seq_length; ++i) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < head_size; ++j) {
                weighted_sum += attention_scores[i] * values[i * d_model + j];
            }
            output[idx * d_model + i] = weighted_sum;
        }
    }
}

Matrix MultiHeadAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask) {
    int seq_length = query.getRows();
    int d_model = query.getCols();

    Matrix output(seq_length, d_model);

    // Llama al kernel usando los datos de los Matrix
    int blockSize = 256;
    int numBlocks = (seq_length + blockSize - 1) / blockSize;
    multiHeadAttentionKernel<<<numBlocks, blockSize>>>(
        query.getData(),
        key.getData(),
        value.getData(),
        output.getData(),
        d_model,
        n_heads,
        seq_length
    );
    cudaDeviceSynchronize();

    return output;
}

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads) 
    : d_model(d_model), n_heads(n_heads) {
    d_k = d_model / n_heads;
    d_v = d_model / n_heads;
    // Inicializar matrices W_Q, W_K, W_V, W_O si las tienes
}

MultiHeadAttention::~MultiHeadAttention() {
    // Liberar memoria si es necesario
    
}