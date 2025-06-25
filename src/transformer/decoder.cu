// File: /cuda-transformer/cuda-transformer/src/transformer/decoder.cu
#include "decoder.cuh"
#include "attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include "utils/cuda_utils.cuh"

Matrix DecoderLayer::forward(const Matrix &input, const Matrix &encoder_output, const Matrix &target_mask, const Matrix *src_mask) {
    // Masked self-attention
    Matrix self_att_output = masked_self_attention.forward(input, input, input, target_mask);
    Matrix norm1_output = norm1.forward(input.add(self_att_output));

    // Encoder-decoder attention
    Matrix enc_dec_att_output = encoder_decoder_attention.forward(norm1_output, encoder_output, encoder_output, src_mask ? *src_mask : Matrix());
    Matrix norm2_output = norm2.forward(norm1_output.add(enc_dec_att_output));

    // Feed-forward
    Matrix ff_output = feed_forward.forward(norm2_output);
    Matrix norm3_output = norm3.forward(norm2_output.add(ff_output));

    return norm3_output;
}