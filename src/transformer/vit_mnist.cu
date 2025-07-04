#include "vit_mnist.cuh"
#include <cmath>
#include <iostream>
#include <random>

// PatchEmbedding Implementation
PatchEmbedding::PatchEmbedding(int patch_size, int embed_dim) 
    : patch_size(patch_size), embed_dim(embed_dim), projection(patch_size * patch_size, embed_dim) {
    // Initialize projection weights randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    std::vector<float> weights(patch_size * patch_size * embed_dim);
    for (auto& w : weights) {
        w = dist(gen);
    }
    projection.copyFromHost(weights);
}

Matrix PatchEmbedding::forward(const Matrix& image) {
    // Convert 28x28 image to patches
    int img_size = 28;
    int num_patches_per_dim = img_size / patch_size;
    int num_patches = num_patches_per_dim * num_patches_per_dim;
    
    printf("[PATCH_EMBEDDING] Input image: %dx%d, patch_size: %d, num_patches: %d\n", 
           image.getRows(), image.getCols(), patch_size, num_patches);
    
    Matrix patches(num_patches, patch_size * patch_size, 0.0f);
    
    // Extract patches (simplified CPU version)
    std::vector<float> image_data(img_size * img_size);
    image.copyToHost(image_data);
    
    printf("[PATCH_EMBEDDING] Image data size: %zu, expected: %d\n", 
           image_data.size(), img_size * img_size);
    
    std::vector<float> patch_data(num_patches * patch_size * patch_size);
    int patch_idx = 0;
    
    for (int i = 0; i < num_patches_per_dim; i++) {
        for (int j = 0; j < num_patches_per_dim; j++) {
            for (int pi = 0; pi < patch_size; pi++) {
                for (int pj = 0; pj < patch_size; pj++) {
                    int img_row = i * patch_size + pi;
                    int img_col = j * patch_size + pj;
                    int patch_pos = patch_idx * patch_size * patch_size + pi * patch_size + pj;
                    patch_data[patch_pos] = image_data[img_row * img_size + img_col];
                }
            }
            patch_idx++;
        }
    }
    
    printf("[PATCH_EMBEDDING] Patch data size: %zu, patches matrix: %dx%d\n", 
           patch_data.size(), patches.getRows(), patches.getCols());
    
    printf("[PATCH_EMBEDDING] About to call copyFromHost - patch_data.size(): %zu, matrix size: %d\n", 
           patch_data.size(), patches.getRows() * patches.getCols());
    
    patches.copyFromHost(patch_data);
    
    // Project patches to embedding dimension
    return patches.multiply(projection);
}

// ViTBlock Implementation
ViTBlock::ViTBlock(int embed_dim, int num_heads) 
    : attention(embed_dim, num_heads), mlp(embed_dim, embed_dim * 4), 
      norm1(embed_dim), norm2(embed_dim) {
}

Matrix ViTBlock::forward(const Matrix& x) {
    // Almacenamos la entrada para la retropropagación
    Matrix input_copy = x;
    
    // Multi-head attention con residual connection
    Matrix attn_out = attention.forward(x, x, x, Matrix());
    Matrix x1 = x.add(attn_out);
    Matrix x1_norm = norm1.forward(x1);
    
    // MLP with residual connection
    Matrix mlp_out = mlp.forward(x1_norm);
    Matrix x2 = x1_norm.add(mlp_out);
    Matrix x2_norm = norm2.forward(x2);
    
    return x2_norm;
}

Matrix ViTBlock::backward(const Matrix& grad_output) {
    // Backward  a través de la normalización 2
    Matrix grad_x2 = norm2.backward(grad_output, Matrix()); // Simplified
    
    // Backward  a través de la conexión residual
    Matrix grad_mlp_out = grad_x2;
    Matrix grad_x1_norm = grad_x2;
    
    // Backward through MLP
    Matrix grad_x1_norm_mlp = mlp.backward(grad_mlp_out, Matrix()); // Simplified
    grad_x1_norm = grad_x1_norm.add(grad_x1_norm_mlp);
    
    // Backward through norm1
    Matrix grad_x1 = norm1.backward(grad_x1_norm, Matrix()); // SImple por ahora
    
    // Backward through residual connection
    Matrix grad_attn_out = grad_x1;
    Matrix grad_x = grad_x1;
    
    // Backward through attention
    Matrix grad_x_attn = attention.backward(grad_attn_out, Matrix(), Matrix(), Matrix()); // Simplified
    grad_x = grad_x.add(grad_x_attn);
    
    return grad_x;
}

void ViTBlock::updateWeights(float learning_rate) {
    attention.updateWeights(learning_rate);
    mlp.updateWeights(learning_rate);
    norm1.updateWeights(learning_rate);
    norm2.updateWeights(learning_rate);
}

// ViTMNIST Implementation
ViTMNIST::ViTMNIST(int patch_size, int embed_dim, int num_heads, int num_layers, int num_classes)
    : patch_embed(patch_size, embed_dim), norm(embed_dim), classifier(embed_dim, num_classes),
      embed_dim(embed_dim), num_classes(num_classes), 
      last_pooled(1, embed_dim, 0.0f), last_normalized(1, embed_dim, 0.0f) {
    
    // Calculate number of patches
    int img_size = 28;
    int num_patches_per_dim = img_size / patch_size;
    num_patches = num_patches_per_dim * num_patches_per_dim;
    
    // Initialize positional embeddings
    pos_embedding = Matrix(num_patches, embed_dim, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    std::vector<float> pos_data(num_patches * embed_dim);
    for (auto& p : pos_data) {
        p = dist(gen);
    }
    pos_embedding.copyFromHost(pos_data);
    
    // Initialize transformer blocks
    blocks.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
        blocks.emplace_back(embed_dim, num_heads);
    }
}

Matrix ViTMNIST::forward(const Matrix& x) {
    // Patch embedding
    Matrix patches = patch_embed.forward(x);
    
    // Add positional embeddings
    Matrix x_with_pos = patches.add(pos_embedding);
    
    // Pass through transformer blocks
    Matrix current = x_with_pos;
    for (auto& block : blocks) {
        current = block.forward(current);
    }
    
    // Layer norm
    last_normalized = norm.forward(current);
    
    // Global average pooling (take mean across patches)
    last_pooled = Matrix(1, embed_dim, 0.0f);
    std::vector<float> norm_data(num_patches * embed_dim);
    last_normalized.copyToHost(norm_data);
    
    std::vector<float> pool_data(embed_dim, 0.0f);
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_patches; j++) {
            sum += norm_data[j * embed_dim + i];
        }
        pool_data[i] = sum / num_patches;
    }
    last_pooled.copyFromHost(pool_data);
    
    // Classification
    return classifier.forward(last_pooled);
}

// --- Métodos de actualización y retropropagación ---

void ViTMNIST::backward(const Matrix& loss_grad) {
    // Backward through classifier
    Matrix grad = classifier.backward(loss_grad, last_pooled); // Usar la variable almacenada
    
    // Expand gradient to match all patches (reverse of global average pooling)
    Matrix expanded_grad(num_patches, embed_dim, 0.0f);
    std::vector<float> grad_data(grad.getRows() * grad.getCols()); // Usar el tamaño real de grad
    grad.copyToHost(grad_data);
    
    // Expand the gradient from 1x128 to 49x128 (replicate for each patch)
    std::vector<float> expanded_data(num_patches * embed_dim);
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < embed_dim; j++) {
            expanded_data[i * embed_dim + j] = grad_data[j] / num_patches; // Divide by num_patches for average pooling backward
        }
    }
    expanded_grad.copyFromHost(expanded_data);
    
    // Backward through layer norm
    Matrix grad_blocks = norm.backward(expanded_grad, Matrix()); // Simplified
    
    // Backward through transformer blocks (in reverse order)
    Matrix current_grad = grad_blocks;
    for (int i = blocks.size() - 1; i >= 0; --i) {
        current_grad = blocks[i].backward(current_grad);
    }
    
    // Note: We don't backward through patch embedding and positional embedding
    // in this simplified version
}

void ViTMNIST::update_weights(float learning_rate) {
    classifier.updateWeights(learning_rate);
    norm.updateWeights(learning_rate);
    for (auto& block : blocks) {
        block.updateWeights(learning_rate);
    }
    // patch_embed weights update would go here in a complete implementation
}

// --- Métodos para PatchEmbedding y otros componentes pueden agregarse si se requiere entrenamiento completo ---
