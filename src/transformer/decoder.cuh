// filepath: cuda-transformer/cuda-transformer/src/transformer/decoder.cuh
#ifndef DECODER_H
#define DECODER_H

#include "../include/common.cuh"
#include "attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include <vector>

class DecoderLayer {
public:
    MultiHeadAttention masked_self_attention;
    MultiHeadAttention encoder_decoder_attention;
    FeedForward feed_forward;
    LayerNorm norm1, norm2, norm3;
    size_t d_model;

    DecoderLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048)
        : masked_self_attention(d_model, n_heads),
          encoder_decoder_attention(d_model, n_heads),
          feed_forward(d_model, d_ff),
          norm1(d_model), norm2(d_model), norm3(d_model),
          d_model(d_model) {}

    Matrix forward(const Matrix &input, const Matrix &encoder_output,
                   const Matrix &target_mask, const Matrix *src_mask = nullptr);
};

#endif // DECODER_H