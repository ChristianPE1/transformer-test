#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda_runtime.h>
#include <vector>

class Matrix
{
private:
    float *data;
    int rows, cols;
    bool on_device;

public:
    __host__ Matrix() : data(nullptr), rows(0), cols(0), on_device(false) {} // Default constructor
    __host__ Matrix(int rows, int cols);
    __host__ Matrix(int rows, int cols, float init_val);
    __host__ ~Matrix();

    // Copy constructor and assignment
    __host__ Matrix(const Matrix &other);
    __host__ Matrix &operator=(const Matrix &other);

    // Basic operations
    __host__ void copyFromHost(const std::vector<float> &hostData);
    __host__ void copyToHost(std::vector<float> &hostData) const;

    // Getters
    __host__ __device__ int getRows() const { return rows; }
    __host__ __device__ int getCols() const { return cols; }
    __host__ __device__ float *getData() const { return data; }

    // Matrix operations
    __host__ Matrix add(const Matrix &other) const;
    __host__ Matrix multiply(const Matrix &other) const;

    // Element access
    __host__ float getElement(int row, int col) const;
    __host__ void setElement(int row, int col, float value);
};

#endif // MATRIX_CUH