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
    float *b1, *b2;

public:
    FeedForward(size_t d_model, size_t d_ff);
    ~FeedForward();

    Matrix forward(const Matrix &input);
    void initializeWeights();
};

#endif // FEED_FORWARD_H