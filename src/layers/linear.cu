// filepath: cuda-transformer/cuda-transformer/src/layers/linear.cu
#include "linear.cuh"
#include "utils/cuda_utils.cuh"

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

    const float *d_input = input.getData();
    const float *d_weights = weights.getData();
    const float *d_bias = bias.getData();

    Matrix output(batch_size, output_dim);
    float *d_output = output.getData();

    linear_forward_kernel<<<batch_size, output_dim>>>(d_input, d_weights, d_bias, d_output, input_dim, output_dim, batch_size);

    cudaDeviceSynchronize();
    return output;
}