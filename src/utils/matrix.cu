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

    // Use CPU implementation for stability
    std::vector<float> host_a(size);
    std::vector<float> host_b(size);
    std::vector<float> host_result(size);
    
    // Copy data to host
    cudaMemcpy(host_a.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b.data(), other.data, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Perform addition on CPU
    for (int i = 0; i < size; i++) {
        host_result[i] = host_a[i] + host_b[i];
    }
    
    // Copy result back to GPU
    cudaMemcpy(result.data, host_result.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    return result;
}

Matrix Matrix::multiply(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    Matrix result(rows, other.cols);
    
    // Use CPU implementation for stability
    std::vector<float> host_a(rows * cols);
    std::vector<float> host_b(other.rows * other.cols);
    std::vector<float> host_result(rows * other.cols, 0.0f);
    
    // Copy data to host
    cudaMemcpy(host_a.data(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b.data(), other.data, other.rows * other.cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Perform matrix multiplication on CPU
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++) {
                sum += host_a[i * cols + k] * host_b[k * other.cols + j];
            }
            host_result[i * other.cols + j] = sum;
        }
    }
    
    // Copy result back to GPU
    cudaMemcpy(result.data, host_result.data(), rows * other.cols * sizeof(float), cudaMemcpyHostToDevice);
    
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