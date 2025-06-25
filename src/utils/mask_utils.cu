// filepath: cuda-transformer/cuda-transformer/src/utils/mask_utils.cu
#include "mask_utils.cuh"
#include <cuda_runtime.h>

__global__ void createPaddingMaskKernel(float *mask, const int *tokens, int seq_len, int pad_token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        mask[idx] = (tokens[idx] == pad_token) ? 0.0f : 1.0f;
    }
}

__global__ void createLookAheadMaskKernel(float *mask, int seq_len) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < seq_len && col < seq_len) {
        mask[row * seq_len + col] = (col <= row) ? 1.0f : 0.0f;
    }
}

__global__ void combineDecoderMasksKernel(float *combined_mask, const float *padding_mask, const float *look_ahead_mask, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        for (int j = 0; j < seq_len; ++j) {
            combined_mask[idx * seq_len + j] = padding_mask[idx] * look_ahead_mask[idx * seq_len + j];
        }
    }
}

void MaskUtils::createPaddingMask(const std::vector<int> &tokens, float *mask, int pad_token) {
    int seq_len = tokens.size();
    int *d_tokens;
    float *d_mask;

    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMalloc(&d_mask, seq_len * sizeof(float));
    cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (seq_len + blockSize - 1) / blockSize;
    createPaddingMaskKernel<<<numBlocks, blockSize>>>(d_mask, d_tokens, seq_len, pad_token);

    cudaMemcpy(mask, d_mask, seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tokens);
    cudaFree(d_mask);
}

void MaskUtils::createLookAheadMask(float *mask, int seq_len) {
    float *d_mask;
    cudaMalloc(&d_mask, seq_len * seq_len * sizeof(float));

    int blockSize = 16;
    int numBlocks = seq_len;
    createLookAheadMaskKernel<<<numBlocks, blockSize>>>(d_mask, seq_len);

    cudaMemcpy(mask, d_mask, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_mask);
}

void MaskUtils::combineDecoderMasks(const std::vector<int> &tokens, float *combined_mask, int pad_token) {
    int seq_len = tokens.size();
    float *padding_mask = new float[seq_len];
    float *look_ahead_mask = new float[seq_len * seq_len];

    createPaddingMask(tokens, padding_mask, pad_token);
    createLookAheadMask(look_ahead_mask, seq_len);

    float *d_combined_mask;
    cudaMalloc(&d_combined_mask, seq_len * seq_len * sizeof(float));

    int blockSize = 256;
    int numBlocks = (seq_len + blockSize - 1) / blockSize;
    combineDecoderMasksKernel<<<numBlocks, blockSize>>>(d_combined_mask, padding_mask, look_ahead_mask, seq_len);

    cudaMemcpy(combined_mask, d_combined_mask, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_combined_mask);
    delete[] padding_mask;
    delete[] look_ahead_mask;
}