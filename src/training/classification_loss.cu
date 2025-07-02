#include "classification_loss.cuh"
#include <cmath>
#include <algorithm>
#include <iostream>

float CrossEntropyLoss::compute_loss(const Matrix& predictions, const std::vector<int>& labels) {
    int num_samples = predictions.getRows();
    int num_classes = predictions.getCols();
    
    std::vector<float> pred_data(num_samples * num_classes);
    predictions.copyToHost(pred_data);
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < num_samples; i++) {
        // Apply softmax to predictions for this sample
        std::vector<float> logits(num_classes);
        for (int j = 0; j < num_classes; j++) {
            logits[j] = pred_data[i * num_classes + j];
        }
        
        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            logits[j] = std::exp(logits[j] - max_logit);
            sum_exp += logits[j];
        }
        
        for (int j = 0; j < num_classes; j++) {
            logits[j] /= sum_exp;
        }
        
        // Compute cross-entropy loss
        int true_label = labels[i];
        if (true_label >= 0 && true_label < num_classes) {
            total_loss -= std::log(std::max(logits[true_label], 1e-8f));
        }
    }
    
    return total_loss / num_samples;
}

Matrix CrossEntropyLoss::compute_gradients(const Matrix& predictions, const std::vector<int>& labels) {
    int num_samples = predictions.getRows();
    int num_classes = predictions.getCols();
    
    std::vector<float> pred_data(num_samples * num_classes);
    predictions.copyToHost(pred_data);
    
    std::vector<float> grad_data(num_samples * num_classes, 0.0f);
    
    for (int i = 0; i < num_samples; i++) {
        // Apply softmax to predictions for this sample
        std::vector<float> logits(num_classes);
        for (int j = 0; j < num_classes; j++) {
            logits[j] = pred_data[i * num_classes + j];
        }
        
        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            logits[j] = std::exp(logits[j] - max_logit);
            sum_exp += logits[j];
        }
        
        for (int j = 0; j < num_classes; j++) {
            logits[j] /= sum_exp;
        }
        
        // Compute gradients
        int true_label = labels[i];
        for (int j = 0; j < num_classes; j++) {
            if (j == true_label) {
                grad_data[i * num_classes + j] = logits[j] - 1.0f;
            } else {
                grad_data[i * num_classes + j] = logits[j];
            }
        }
    }
    
    Matrix gradients(num_samples, num_classes);
    gradients.copyFromHost(grad_data);
    return gradients;
}
