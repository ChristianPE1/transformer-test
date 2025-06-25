#ifndef EMBEDDINGS_CUH
#define EMBEDDINGS_CUH

#include <cuda_runtime.h>
#include <vector>
#include "../utils/matrix.cuh"

class Embedding
{
private:
    size_t vocab_size;
    size_t d_model;
    float *weights; // Device pointer

public:
    Embedding(size_t vocab_size, size_t d_model);
    ~Embedding();

    void initializeWeights();
    Matrix forward(const std::vector<int> &input_tokens);
    void updateWeights(const Matrix& gradients, float learning_rate, const std::vector<int>& tokens);
};

class PositionalEncoding
{
private:
    size_t d_model;
    size_t max_len;
    float *pos_encodings; // Device pointer

public:
    PositionalEncoding(size_t d_model, size_t max_len = 512);
    ~PositionalEncoding();

    void initializeEncodings();
    Matrix getEncoding(int seq_len);
};

#endif // EMBEDDINGS_CUH