// filepath: cuda-transformer/cuda-transformer/src/layers/feed_forward.cuh
#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

class FeedForward {
private:
    size_t d_model;
    size_t d_ff;
    Matrix W1, W2;
    Matrix grad_W1, grad_W2;  // Gradients for weight matrices
    float *b1, *b2;
    float *grad_b1, *grad_b2; // Gradients for biases

public:
    FeedForward(size_t d_model, size_t d_ff);
    ~FeedForward();

    Matrix forward(const Matrix &input);
    Matrix backward(const Matrix &grad_output, const Matrix &input);
    void updateWeights(float learning_rate);
    void initializeWeights();
};

#endif // FEED_FORWARD_H