#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "data/dataset.cuh"
#include "data/vocab.cuh"
#include "transformer/transformer.cuh"
#include "transformer/vision_transformer.cuh"
#include "utils/matrix.cuh"
#include "training/loss.cuh"
#include "training/optimizer.cuh"
#include "training/trainer.cuh"
#include "data/mnist_loader.cuh"

Matrix compute_loss(const Matrix &predictions, const Matrix &labels) {
    // Implementación de cálculo de pérdida
    Matrix loss;
    // ...
    return loss;
}

int main()
{
    try
    {
        std::cout << "=== Vision Transformer for MNIST ===" << std::endl;

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
        MNISTLoader loader;
        MNISTData mnist_data = loader.load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

        // Initialize Vision Transformer
        size_t d_model = 128;
        size_t n_heads = 8;
        size_t n_layers = 6;
        size_t d_ff = 512;
        size_t patch_size = 4;
        size_t num_classes = 10;

        VisionTransformer vit_model(d_model, n_heads, n_layers, d_ff);
        vit_model.initialize(patch_size, num_classes);

        // Training loop
        for (size_t epoch = 0; epoch < 10; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "..." << std::endl;

            for (const auto &image : mnist_data.images) {
                // Convert image to Matrix
                Matrix input(28, 28, 0.0f);
                input.copyFromHost(image);

                Matrix predictions = vit_model.forward(input);

                // Convert labels to Matrix
                std::vector<float> labels_vector(mnist_data.labels.begin(), mnist_data.labels.end());
                Matrix labels(1, mnist_data.labels.size(), 0.0f);
                labels.copyFromHost(labels_vector);

                auto loss = compute_loss(predictions, labels);

                vit_model.backward(loss);
                vit_model.update_weights();
            }
        }

        std::cout << "Training complete." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}