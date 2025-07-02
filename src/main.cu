#include <iostream>
#include <cuda_runtime.h>
#include "transformer/vit_mnist.cuh"
#include "data/mnist_loader.cuh"
#include "training/classification_loss.cuh"

int main()
{
    try
    {
        std::cout << "=== Vision Transformer for MNIST Classification ===" << std::endl;

        // Verify CUDA
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cout << "CUDA devices found: " << deviceCount << std::endl;

        if (deviceCount == 0)
        {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        // Load MNIST dataset
        std::cout << "Loading MNIST dataset..." << std::endl;
        MNISTLoader loader;
        MNISTData mnist_data = loader.load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        
        std::cout << "Loaded " << mnist_data.images.size() << " images and " 
                  << mnist_data.labels.size() << " labels" << std::endl;

        // Initialize Vision Transformer
        int patch_size = 4;  // 28/4 = 7 patches per dimension, 49 total patches
        int embed_dim = 128;
        int num_heads = 8;
        int num_layers = 6;
        int num_classes = 10;

        ViTMNIST vit_model(patch_size, embed_dim, num_heads, num_layers, num_classes);
        std::cout << "Vision Transformer initialized" << std::endl;

        // Training parameters
        int num_epochs = 5;
        int batch_size = 32;
        float learning_rate = 0.001f;

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\nEpoch " << (epoch + 1) << "/" << num_epochs << std::endl;
            
            float epoch_loss = 0.0f;
            int num_batches = 0;
            
            // Process images in batches (limit to 100 for testing)
            for (size_t i = 0; i < std::min((size_t)100, mnist_data.images.size()); i += batch_size) {
                size_t batch_end = std::min(i + batch_size, mnist_data.images.size());
                
                // Process each image in the batch
                for (size_t j = i; j < batch_end; ++j) {
                    // Convert image to Matrix
                    Matrix image(28, 28, 0.0f);
                    image.copyFromHost(mnist_data.images[j]);
                    
                    // Forward pass
                    Matrix predictions = vit_model.forward(image);
                    
                    // Compute loss
                    std::vector<int> single_label = {mnist_data.labels[j]};
                    float loss = CrossEntropyLoss::compute_loss(predictions, single_label);
                    epoch_loss += loss;
                    
                    // Compute gradients
                    Matrix loss_grad = CrossEntropyLoss::compute_gradients(predictions, single_label);
                    
                    // Backward pass (simplified)
                    vit_model.backward(loss_grad);
                    vit_model.update_weights(learning_rate);
                }
                
                num_batches++;
                if (num_batches % 10 == 0) {
                    std::cout << "Processed " << (batch_end) << " samples, avg loss: " 
                              << (epoch_loss / (batch_end - i)) << std::endl;
                }
            }
            
            std::cout << "Epoch " << (epoch + 1) << " completed. Average loss: " 
                      << (epoch_loss / std::min((size_t)100, mnist_data.images.size())) << std::endl;
        }

        std::cout << "\nTraining completed!" << std::endl;

    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}