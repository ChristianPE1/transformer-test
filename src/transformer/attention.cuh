// attention.cuh
#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t n_heads);
    ~MultiHeadAttention();

    Matrix forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask = Matrix());

private:
    size_t d_model;
    size_t n_heads;
    size_t d_k;
    size_t d_v;

    Matrix W_Q; // Weight matrix for queries
    Matrix W_K; // Weight matrix for keys
    Matrix W_V; // Weight matrix for values
    Matrix W_O; // Output weight matrix

    void splitHeads(const Matrix &input, Matrix &output);
    void combineHeads(const Matrix &input, Matrix &output);
    Matrix scaledDotProductAttention(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask);
};

#endif // ATTENTION_CUH