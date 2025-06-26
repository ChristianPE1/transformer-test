#include "utils/matrix.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <algorithm>

__global__ void matrixAddKernel(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] + b[idx];
    }
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), on_device(true)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemset(data, 0, rows * cols * sizeof(float));
}

Matrix::Matrix(int rows, int cols, float init_val) : rows(rows), cols(cols), on_device(true)
{
    cudaMalloc(&data, rows * cols * sizeof(float));

    std::vector<float> host_data(rows * cols, init_val);
    cudaMemcpy(data, host_data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

Matrix::~Matrix()
{
    if (data && on_device)
    {
        cudaFree(data);
    }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols), on_device(true)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (this != &other)
    {
        if (data && on_device)
        {
            cudaFree(data);
        }

        rows = other.rows;
        cols = other.cols;
        on_device = true;

        cudaMalloc(&data, rows * cols * sizeof(float));
        cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

Matrix Matrix::add(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::runtime_error("Matrix dimensions don't match for addition");
    }

    Matrix result(rows, cols);
    int size = rows * cols;

    // DEBUG: Check input matrices before addition
    std::vector<float> debug_a(std::min(10, size));
    std::vector<float> debug_b(std::min(10, size));
    
    cudaMemcpy(debug_a.data(), data, debug_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_b.data(), other.data, debug_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("[MATRIX_ADD_DEBUG] Matrix A first 5 values: ");
    for (int i = 0; i < std::min(5, (int)debug_a.size()); i++) {
        printf("%.6f ", debug_a[i]);
    }
    printf("\n");
    
    printf("[MATRIX_ADD_DEBUG] Matrix B first 5 values: ");
    for (int i = 0; i < std::min(5, (int)debug_b.size()); i++) {
        printf("%.6f ", debug_b[i]);
    }
    printf("\n");

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    matrixAddKernel<<<numBlocks, blockSize>>>(data, other.data, result.data, size);
    cudaDeviceSynchronize();

    // DEBUG: Check result after addition
    std::vector<float> debug_result(std::min(10, size));
    cudaMemcpy(debug_result.data(), result.data, debug_result.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("[MATRIX_ADD_DEBUG] Result first 5 values: ");
    for (int i = 0; i < std::min(5, (int)debug_result.size()); i++) {
        printf("%.6f ", debug_result[i]);
    }
    printf("\n");

    return result;
}

Matrix Matrix::multiply(const Matrix &other) const
{
    // Implementación simplificada
    Matrix result(rows, other.cols);
    return result;
}

void Matrix::copyFromHost(const std::vector<float> &hostData)
{
    if (hostData.size() != rows * cols)
    {
        throw std::runtime_error("Host data size doesn't match matrix size");
    }

    cudaMemcpy(data, hostData.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::copyToHost(std::vector<float> &hostData) const
{
    hostData.resize(rows * cols);
    cudaMemcpy(hostData.data(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

float Matrix::getElement(int row, int col) const
{
    float value;
    cudaMemcpy(&value, &data[row * cols + col], sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

void Matrix::setElement(int row, int col, float value)
{
    cudaMemcpy(&data[row * cols + col], &value, sizeof(float), cudaMemcpyHostToDevice);
}