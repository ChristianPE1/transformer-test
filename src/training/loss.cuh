// filepath: cuda-transformer/cuda-transformer/src/training/loss.cuh
#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <iomanip>
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
        
        // SIMPLIFIED CROSS-ENTROPY - more stable and faster
        for (int i = 0; i < batch_size; i++) {
            int target_class = static_cast<int>(targets.getElement(i, 0));
            if (target_class >= 0 && target_class < num_classes) {
                // Get prediction for target class
                float prediction = predictions.getElement(i, target_class);
                
                // Simple stabilized cross-entropy: -log(softmax(pred))
                // Find max for numerical stability
                float max_val = predictions.getElement(i, 0);
                for (int j = 1; j < num_classes; j++) {
                    max_val = std::max(max_val, predictions.getElement(i, j));
                }
                
                // Compute log-sum-exp
                double sum_exp = 0.0;
                for (int j = 0; j < num_classes; j++) {
                    sum_exp += exp(predictions.getElement(i, j) - max_val);
                }
                
                // Cross-entropy loss: -log(softmax(target))
                double log_prob = (prediction - max_val) - log(sum_exp + 1e-8);
                total_loss -= log_prob;
            }
        }
        
        double avg_loss = total_loss / batch_size;
        
        // Clamp to reasonable range
        if (!std::isfinite(avg_loss) || avg_loss > 20.0) {
            avg_loss = 20.0;
        }
        
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
            
            // Label smoothing parameters - REDUCIDO para acelerar aprendizaje inicial
            const float label_smoothing = 0.05f;  // Reducido de 0.1 a 0.05
            const float true_prob = 1.0f - label_smoothing;
            const float smooth_prob = label_smoothing / (num_classes - 1);
            
            // Calcular gradientes del softmax con label smoothing
            for (int j = 0; j < num_classes; j++) {
                float softmax_val = static_cast<float>(exp_vals[j] / sum_exp);
                
                // Clamp softmax para evitar valores extremos
                softmax_val = std::max(1e-7f, std::min(1.0f - 1e-7f, softmax_val));
                
                float target_prob;
                if (j == target_class) {
                    target_prob = true_prob;
                } else {
                    target_prob = smooth_prob;
                    
                    // Penalización extra para <eos> (token 3) en posiciones tempranas
                    if (j == 3 && i < 2) {  // Penalizar <eos> en las primeras 2 posiciones
                        target_prob *= 0.1f;  // Reducir probabilidad objetivo por 10x
                    }
                }
                
                float gradient = softmax_val - target_prob;
                
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
    
    // Función auxiliar para calcular penalización de EOS temprano
    double calculateEOSPenalty(const Matrix& predictions, const Matrix& targets) {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        double penalty = 0.0;
        
        for (int i = 0; i < batch_size; i++) {
            // Solo penalizar en las primeras 3 posiciones de la secuencia
            if (i < 3) {
                float eos_logit = predictions.getElement(i, 3); // Token 3 es <eos>
                
                // Si <eos> tiene alta probabilidad en posición temprana, penalizar
                if (eos_logit > 0.0f) {
                    penalty += eos_logit * 2.0f; // Penalización proporcional al logit
                }
            }
        }
        
        return penalty / batch_size;
    }
};
#endif // LOSS_H
