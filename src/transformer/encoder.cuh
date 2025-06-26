#pragma once
#ifndef ENCODER_H
#define ENCODER_H

#include "../include/common.cuh"
#include "attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include <vector>

class EncoderLayer {
private:
    MultiHeadAttention self_attention;
    FeedForward feed_forward;
    LayerNorm norm1, norm2;

public:
    EncoderLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048)
        : self_attention(d_model, n_heads), 
          feed_forward(d_model, d_ff),
          norm1(d_model), 
          norm2(d_model) {}

    Matrix forward(const Matrix &input, const Matrix *src_mask = nullptr) {
        Matrix self_att_output = self_attention.forward(input, input, input, src_mask ? *src_mask : Matrix());
        Matrix norm1_output = norm1.forward(input.add(self_att_output));
        Matrix ff_output = feed_forward.forward(norm1_output);
        Matrix norm2_output = norm2.forward(norm1_output.add(ff_output));
        return norm2_output;
    }
};

// Agrega la clase Encoder
class Encoder {
private:
    std::vector<EncoderLayer> layers;
    size_t n_layers;

public:
    Encoder(size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff = 2048);
    void forward(const Matrix &input, const Matrix &src_mask, Matrix &output);
};

//#endif // ENCODER_H