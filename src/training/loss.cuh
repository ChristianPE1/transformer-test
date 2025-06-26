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
        
        // Calculate prediction diversity (how spread out the predictions are)
        float pred_variance = 0.0f;
        float pred_mean = 0.0f;
        for (size_t i = 0; i < pred_check.size(); ++i) {
            pred_mean += pred_check[i];
        }
        pred_mean /= pred_check.size();
        
        for (size_t i = 0; i < pred_check.size(); ++i) {
            pred_variance += (pred_check[i] - pred_mean) * (pred_check[i] - pred_mean);
        }
        pred_variance /= pred_check.size();
        
        std::cout << " Var:" << std::setprecision(3) << pred_variance;
        std::cout << std::endl;
        
        if (has_nan || has_inf || max_pred > 50.0f || min_pred < -50.0f) {
            std::cout << "[LOSS] ERROR: Invalid predictions detected, returning large loss" << std::endl;
            return 100.0; // Return a large but finite loss
        }
        
        // Calcular cross-entropy real con mejor estabilización numérica y label smoothing
        const float label_smoothing = 0.1f;  // 10% label smoothing
        const float true_prob = 1.0f - label_smoothing;
        const float smooth_prob = label_smoothing / (num_classes - 1);
        
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
                
                // Calcular loss con label smoothing
                double batch_loss = 0.0;
                for (int j = 0; j < num_classes; j++) {
                    double log_prob = (predictions.getElement(i, j) - max_val) - log(sum_exp);
                    
                    double target_prob;
                    if (j == target_class) {
                        target_prob = true_prob;
                    } else {
                        target_prob = smooth_prob;
                        
                        // Penalización extra para <eos> (token 3) en posiciones tempranas
                        if (j == 3 && i < 2) {
                            target_prob *= 0.1f;  // Reducir probabilidad por 10x
                        }
                    }
                    
                    if (std::isfinite(log_prob)) {
                        batch_loss -= target_prob * log_prob;
                    }
                }
                
                // Verificar que el resultado sea finito
                if (std::isfinite(batch_loss)) {
                    total_loss += batch_loss;
                } else {
                    std::cout << "[LOSS] WARNING: Non-finite loss for sample " << i << std::endl;
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
            
            // Label smoothing parameters
            const float label_smoothing = 0.1f;  // 10% label smoothing
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
