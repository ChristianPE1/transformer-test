// filepath: cuda-transformer/cuda-transformer/src/training/loss.cu
#include "loss.cuh"
#include "utils/cuda_utils.cuh"

__device__ float crossEntropyLoss(const float* predictions, const int* targets, int num_classes, int batch_size) {
    float loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        int target = targets[i];
        loss -= logf(predictions[i * num_classes + target] + 1e-10); // Adding epsilon to avoid log(0)
    }
    return loss / batch_size;
}

__global__ void computeLossKernel(const float* predictions, const int* targets, float* loss, int num_classes, int batch_size) {
    float loss_value = crossEntropyLoss(predictions, targets, num_classes, batch_size);
    *loss = loss_value;
}

void Loss::calculateCrossEntropy(const float* predictions, const int* targets, float* loss, int num_classes, int batch_size) {
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    
    computeLossKernel<<<1, 1>>>(predictions, targets, d_loss, num_classes, batch_size);
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_loss);
}