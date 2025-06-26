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
        
        // Debug predictions
        std::vector<float> pred_check;
        predictions.copyToHost(pred_check);
        
        bool has_nan = false, has_inf = false;
        float min_pred = pred_check[0], max_pred = pred_check[0];
        for (size_t i = 0; i < pred_check.size(); ++i) {
            if (std::isnan(pred_check[i])) has_nan = true;
            if (std::isinf(pred_check[i])) has_inf = true;
            min_pred = std::min(min_pred, pred_check[i]);
            max_pred = std::max(max_pred, pred_check[i]);
        }
        
        std::cout << "[LOSS] Predictions range: [" << min_pred << ", " << max_pred << "]";
        if (has_nan) std::cout << " [HAS NaN!]";
        if (has_inf) std::cout << " [HAS INF!]";
        std::cout << std::endl;
        
        if (has_nan || has_inf || max_pred > 50.0f || min_pred < -50.0f) {
            std::cout << "[LOSS] ERROR: Invalid predictions detected, returning large loss" << std::endl;
            return 100.0; // Return a large but finite loss
        }
        
        // Calcular cross-entropy real con mejor estabilización numérica
        for (int i = 0; i < batch_size; i++) {
            int target_class = static_cast<int>(targets.getElement(i, 0));
            if (target_class >= 0 && target_class < num_classes) {
                // Encontrar el valor máximo para estabilización numérica
                float max_val = predictions.getElement(i, 0);
                for (int j = 1; j < num_classes; j++) {
                    max_val = std::max(max_val, predictions.getElement(i, j));
                }
                
                // Calcular suma de exponenciales estabilizada
                double sum_exp = 0.0;
                for (int j = 0; j < num_classes; j++) {
                    double exp_val = exp(predictions.getElement(i, j) - max_val);
                    if (std::isfinite(exp_val)) {
                        sum_exp += exp_val;
                    }
                }
                
                // Evitar log(0) y valores extremos
                if (sum_exp <= 1e-10) {
                    sum_exp = 1e-10;
                }
                
                // Calcular log-softmax estabilizado
                double target_logit = predictions.getElement(i, target_class) - max_val;
                double log_softmax = target_logit - log(sum_exp);
                
                // Verificar que el resultado sea finito
                if (std::isfinite(log_softmax)) {
                    total_loss -= log_softmax;
                } else {
                    std::cout << "[LOSS] WARNING: Non-finite log_softmax for sample " << i << std::endl;
                    total_loss += 10.0; // Penalizar con una pérdida alta pero finita
                }
            }
        }
        
        double avg_loss = total_loss / batch_size;
        
        std::cout << "[LOSS] Calculated loss: " << avg_loss;
        if (!std::isfinite(avg_loss)) {
            std::cout << " [NOT FINITE - CLAMPING]";
            avg_loss = 100.0; // Clamp to a large but finite value
        }
        std::cout << std::endl;
        
        return avg_loss;
    }

    Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        Matrix grad(batch_size, num_classes, 0.0f);
        
        // Calcular gradiente real de cross-entropy con softmax estabilizado
        for (int i = 0; i < batch_size; i++) {
            // Calcular softmax estabilizado para la fila i
            float max_val = predictions.getElement(i, 0);
            for (int j = 1; j < num_classes; j++) {
                max_val = std::max(max_val, predictions.getElement(i, j));
            }
            
            // Calcular suma de exponenciales de forma estable
            double sum_exp = 0.0;
            std::vector<double> exp_vals(num_classes);
            for (int j = 0; j < num_classes; j++) {
                double exp_val = exp(predictions.getElement(i, j) - max_val);
                if (std::isfinite(exp_val)) {
                    exp_vals[j] = exp_val;
                    sum_exp += exp_val;
                } else {
                    exp_vals[j] = 0.0;
                }
            }
            
            // Evitar división por cero
            if (sum_exp <= 1e-10) {
                sum_exp = 1e-10;
            }
            
            int target_class = static_cast<int>(targets.getElement(i, 0));
            
            // Calcular gradientes del softmax
            for (int j = 0; j < num_classes; j++) {
                float softmax_val = static_cast<float>(exp_vals[j] / sum_exp);
                
                // Clamp softmax para evitar valores extremos
                softmax_val = std::max(1e-7f, std::min(1.0f - 1e-7f, softmax_val));
                
                float gradient = softmax_val;
                if (j == target_class) {
                    gradient -= 1.0f;
                }
                
                // Clamp gradiente para evitar explosión
                gradient = std::max(-10.0f, std::min(10.0f, gradient));
                
                grad.setElement(i, j, gradient / batch_size);
            }
        }
        
        // Debug gradientes
        std::vector<float> grad_check;
        grad.copyToHost(grad_check);
        
        bool has_nan = false;
        float grad_sum = 0.0f;
        for (size_t i = 0; i < grad_check.size(); ++i) {
            if (std::isnan(grad_check[i]) || std::isinf(grad_check[i])) {
                has_nan = true;
                grad_check[i] = 0.0f; // Replace NaN/Inf with 0
            }
            grad_sum += abs(grad_check[i]);
        }
        
        if (has_nan) {
            std::cout << "[LOSS] WARNING: NaN/Inf in gradients, cleaned" << std::endl;
            grad.copyFromHost(grad_check);
        }
        
        std::cout << "[LOSS] Gradient sum: " << grad_sum << std::endl;
        
        return grad;
    }
};
#endif // LOSS_H
