#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <vector>
#include "embeddings.cuh"
#include "encoder.cuh"
#include "decoder.cuh"
#include "../utils/matrix.cuh"
#include "../layers/linear.cuh"

// Forward declarations para evitar dependencias circulares
class EncoderLayer;
class DecoderLayer;
class Linear;

class Transformer
{
private:
    Embedding input_embedding;
    Embedding target_embedding;
    PositionalEncoding pos_encoding;
    Encoder encoder;
    std::vector<DecoderLayer> decoder_layers;
    Linear output_projection;
    
    size_t d_model;
    size_t n_layers;
    size_t input_vocab_size;
    size_t target_vocab_size;
    
    // Store tokens from last forward pass for gradient updates
    std::vector<int> last_target_tokens;

public:
    Transformer(size_t input_vocab_size, size_t target_vocab_size,
                size_t d_model = 512, size_t n_heads = 8,
                size_t n_layers = 6, size_t d_ff = 2048);    Matrix encode(const std::vector<int> &input_tokens);
    Matrix decode(const std::vector<int> &target_tokens, const Matrix &encoder_output);
    Matrix createCausalMask(int seq_len);
    Matrix forward(const std::vector<int> &source_tokens, const std::vector<int> &target_tokens);
    std::vector<int> generate(const std::vector<int> &source_tokens,
                              int sos_token = 2, int eos_token = 3,
                              size_t max_length = 50);
    
    // Training methods
    void backward(const Matrix& grad_output, float learning_rate);
    void updateWeights(const Matrix& gradients, float learning_rate);
    void updateTargetEmbeddings(const Matrix& gradients, float learning_rate);
    
    size_t getTargetVocabSize() const { return target_vocab_size; }
};

#endif // TRANSFORMER_CUH