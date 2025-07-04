#ifndef VIT_MNIST_CUH
#define VIT_MNIST_CUH

#include "../utils/matrix.cuh"
#include "../layers/linear.cuh"
#include "attention.cuh"
#include "../layers/layer_norm.cuh"
#include "../layers/feed_forward.cuh"

class PatchEmbedding {
private:
    int patch_size;
    int embed_dim;
    Matrix projection; // Proyección lineal para los parches

public:
    PatchEmbedding(int patch_size, int embed_dim);
    Matrix forward(const Matrix& image); // Convierte la imagen 28x28 en parches
};

class ViTBlock {
private:
    MultiHeadAttention attention;
    FeedForward mlp;
    LayerNorm norm1, norm2;

public:
    ViTBlock(int embed_dim, int num_heads);
    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& grad_output);
    void updateWeights(float learning_rate);
};

class ViTMNIST {
private:
    PatchEmbedding patch_embed;
    std::vector<ViTBlock> blocks;
    LayerNorm norm;
    Linear classifier; // Capa de clasificación final
    Matrix pos_embedding; // Embeddings posicionales
    int num_patches;
    int embed_dim;
    int num_classes;
    
    // Variables para almacenar estados intermedios durante forward pass
    Matrix last_pooled; // Para usar en backward
    Matrix last_normalized; // Para usar en backward

public:
    ViTMNIST(int patch_size = 4, int embed_dim = 128, int num_heads = 8, int num_layers = 6, int num_classes = 10);
    Matrix forward(const Matrix& x);
    void backward(const Matrix& loss_grad);
    void update_weights(float learning_rate = 0.001f);
};

#endif // VIT_MNIST_CUH
