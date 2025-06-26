#pragma once
#ifndef ENCODER_H
#define ENCODER_H

#include "../include/common.cuh"
#include "attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include <vector>

class EncoderLayer {
public:
    MultiHeadAttention self_attention;
    FeedForward feed_forward;
    LayerNorm norm1, norm2;

    EncoderLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048)
        : self_attention(d_model, n_heads), 
          feed_forward(d_model, d_ff),
          norm1(d_model), 
          norm2(d_model) {}

    Matrix forward(const Matrix &input, const Matrix *src_mask = nullptr) {
        printf("[ENCODER_LAYER] Starting layer with input stats: ");
        // Quick debug of input
        std::vector<float> debug_input(std::min(10, (int)(input.getRows() * input.getCols())));
        input.copyToHost(debug_input);
        int non_zero = 0;
        for (float val : debug_input) if (val != 0.0f) non_zero++;
        printf("%d/%d non-zero\n", non_zero, (int)debug_input.size());
        
        Matrix self_att_output = self_attention.forward(input, input, input, src_mask ? *src_mask : Matrix());
        
        printf("[ENCODER_LAYER] After self-attention: ");
        std::vector<float> debug_att(std::min(10, (int)(self_att_output.getRows() * self_att_output.getCols())));
        self_att_output.copyToHost(debug_att);
        non_zero = 0;
        for (float val : debug_att) if (val != 0.0f) non_zero++;
        printf("%d/%d non-zero\n", non_zero, (int)debug_att.size());
        
        Matrix residual1 = input.add(self_att_output);
        Matrix norm1_output = norm1.forward(residual1);
        
        printf("[ENCODER_LAYER] After norm1: ");
        std::vector<float> debug_norm1(std::min(10, (int)(norm1_output.getRows() * norm1_output.getCols())));
        norm1_output.copyToHost(debug_norm1);
        non_zero = 0;
        for (float val : debug_norm1) if (val != 0.0f) non_zero++;
        printf("%d/%d non-zero\n", non_zero, (int)debug_norm1.size());
        
        Matrix ff_output = feed_forward.forward(norm1_output);
        
        printf("[ENCODER_LAYER] After feedforward: ");
        std::vector<float> debug_ff(std::min(10, (int)(ff_output.getRows() * ff_output.getCols())));
        ff_output.copyToHost(debug_ff);
        non_zero = 0;
        for (float val : debug_ff) if (val != 0.0f) non_zero++;
        printf("%d/%d non-zero\n", non_zero, (int)debug_ff.size());
        
        Matrix residual2 = norm1_output.add(ff_output);
        Matrix norm2_output = norm2.forward(residual2);
        
        printf("[ENCODER_LAYER] Final output: ");
        std::vector<float> debug_final(std::min(10, (int)(norm2_output.getRows() * norm2_output.getCols())));
        norm2_output.copyToHost(debug_final);
        non_zero = 0;
        for (float val : debug_final) if (val != 0.0f) non_zero++;
        printf("%d/%d non-zero\n", non_zero, (int)debug_final.size());
        
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

#endif // ENCODER_H