// attention.cuh
#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include <cuda_runtime.h>
#include "utils/matrix.cuh"
#include <cstdlib>
#include <vector>

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t n_heads);
    ~MultiHeadAttention();

    Matrix forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask = Matrix());
    
    // Backward pass for training
    void backward(const Matrix &grad_output, Matrix &grad_query, Matrix &grad_key, Matrix &grad_value);
    
    // Update weights
    void updateWeights(float learning_rate);

private:
    size_t d_model;
    size_t n_heads;
    size_t d_k;
    size_t d_v;

    Matrix W_Q; // Weight matrix for queries
    Matrix W_K; // Weight matrix for keys
    Matrix W_V; // Weight matrix for values
    Matrix W_O; // Output weight matrix
    
    // Gradients for weights
    Matrix grad_W_Q;
    Matrix grad_W_K;
    Matrix grad_W_V;
    Matrix grad_W_O;
    
    // Store attention weights for backward pass
    std::vector<float> last_attention_weights;
};

#endif // ATTENTION_CUH