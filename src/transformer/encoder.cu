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
    Matrix current_input = input;

    for (size_t i = 0; i < n_layers; ++i) {
        Matrix layer_output = layers[i].forward(current_input, &src_mask);

        // Validar dimensiones antes de sumar
        if (current_input.getRows() != layer_output.getRows() || current_input.getCols() != layer_output.getCols()) {
            throw std::runtime_error("Matrix dimensions don't match for addition in Encoder layer " + std::to_string(i));
        }

        current_input = layer_output;
    }

    output = current_input;
}