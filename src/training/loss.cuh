// filepath: cuda-transformer/cuda-transformer/src/training/loss.cuh
#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
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
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        
        double total_loss = 0.0;
        
        // Calcular cross-entropy real
        for (int i = 0; i < batch_size; i++) {
            int target_class = static_cast<int>(targets.getElement(i, 0));
            if (target_class >= 0 && target_class < num_classes) {
                float pred = predictions.getElement(i, target_class);
                // Aplicar softmax estabilizado y cross-entropy
                float max_val = predictions.getElement(i, 0);
                for (int j = 1; j < num_classes; j++) {
                    max_val = fmaxf(max_val, predictions.getElement(i, j));
                }
                
                float sum_exp = 0.0f;
                for (int j = 0; j < num_classes; j++) {
                    sum_exp += expf(predictions.getElement(i, j) - max_val);
                }
                
                float log_softmax = (pred - max_val) - logf(sum_exp);
                total_loss -= log_softmax;
            }
        }
        
        return total_loss / batch_size;
    }

    Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        Matrix grad(batch_size, num_classes, 0.0f);
        
        // Calcular gradiente real de cross-entropy con softmax
        for (int i = 0; i < batch_size; i++) {
            // Calcular softmax para la fila i
            float max_val = predictions.getElement(i, 0);
            for (int j = 1; j < num_classes; j++) {
                max_val = fmaxf(max_val, predictions.getElement(i, j));
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                sum_exp += expf(predictions.getElement(i, j) - max_val);
            }
            
            int target_class = static_cast<int>(targets.getElement(i, 0));
            
            for (int j = 0; j < num_classes; j++) {
                float softmax_val = expf(predictions.getElement(i, j) - max_val) / sum_exp;
                float gradient = softmax_val;
                if (j == target_class) {
                    gradient -= 1.0f;
                }
                grad.setElement(i, j, gradient / batch_size);
            }
        }
        
        return grad;
    }
};
#endif // LOSS_H
