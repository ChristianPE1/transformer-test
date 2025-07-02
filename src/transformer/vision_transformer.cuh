#ifndef VISION_TRANSFORMER_CUH
#define VISION_TRANSFORMER_CUH

#include "encoder.cuh"
#include "../utils/matrix.cuh"

class VisionTransformer {
private:
    Encoder encoder;
    size_t d_model;

public:
    VisionTransformer(size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
        : encoder(d_model, n_heads, n_layers, d_ff), d_model(d_model) {}

    Matrix forward(const Matrix &input) {
        Matrix output;
        Matrix src_mask(input.getRows(), input.getCols(), 1.0f); // No masking for MNIST
        encoder.forward(input, src_mask, output);
        return output;
    }
};

#endif // VISION_TRANSFORMER_CUH
