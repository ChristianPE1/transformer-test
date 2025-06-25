#include "encoder.cuh"
#include "attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include "../utils/cuda_utils.cuh"

// Constructor de Encoder
Encoder::Encoder(size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
    : n_layers(n_layers) {
    layers.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        layers.emplace_back(d_model, n_heads, d_ff);
    }
}

void Encoder::forward(const Matrix &input, const Matrix &src_mask, Matrix &output) {
    // Procesa las capas secuencialmente en CPU
    Matrix current_input = input;
    
    for (size_t i = 0; i < n_layers; ++i) {
        current_input = layers[i].forward(current_input, &src_mask);
    }
    
    // Copia el resultado final al output
    output = current_input;
}