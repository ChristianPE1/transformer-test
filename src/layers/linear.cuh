// filepath: cuda-transformer/cuda-transformer/src/layers/linear.cuh
#ifndef LINEAR_H
#define LINEAR_H

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

class Linear {
private:
    Matrix weights; // Weight matrix
    Matrix bias;    // Bias vector
    size_t input_dim;  // Input dimension
    size_t output_dim; // Output dimension
    
    // Store gradients for weight updates
    Matrix stored_grad_weights;
    Matrix stored_grad_bias;

public:
    // Constructor
    Linear(size_t input_dim, size_t output_dim);

    // Forward pass
    Matrix forward(const Matrix &input);
    
    // Backward pass
    Matrix backward(const Matrix &grad_output, const Matrix &input);
    
    // Update weights using stored gradients
    void updateWeights(float learning_rate);

    // Method to initialize weights and bias
    void initialize();

    // Destructor
    ~Linear();
};

#endif // LINEAR_H