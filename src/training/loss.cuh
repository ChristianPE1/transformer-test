// filepath: cuda-transformer/cuda-transformer/src/training/loss.cuh
#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>
#include <iostream>
#include "utils/matrix.cuh"

class Loss {
public:
    virtual double forward(const Matrix& predictions, const Matrix& targets) = 0;
    virtual Matrix backward(const Matrix& predictions, const Matrix& targets) = 0;

    void calculateCrossEntropy(const float* predictions, const int* targets, float* loss, int num_classes, int batch_size);
};

class CrossEntropyLoss : public Loss {
public:
    double forward(const Matrix& predictions, const Matrix& targets) override {
          // VERSIÓN RÁPIDA: Solo calcula loss dummy por ahora
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        
        // Loss simulado basado en el tamaño
        double dummy_loss = batch_size * 0.1 + num_classes * 0.001;
        
        std::cout << " [FAST-LOSS] batch:" << batch_size 
                  << " classes:" << num_classes 
                  << " loss:" << dummy_loss;
                  
        return dummy_loss;
    }

    Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        // Gradiente dummy por ahora
        Matrix grad(predictions.getRows(), predictions.getCols(), 0.01f);
        return grad;
    }
};
#endif // LOSS_H
