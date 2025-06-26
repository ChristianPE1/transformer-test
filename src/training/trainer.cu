// filepath: cuda-transformer/cuda-transformer/src/training/trainer.cu
#include "trainer.cuh"
#include <iostream>
#include <iomanip>

// Convierte un vector de índices a una matriz one-hot (batch_size x num_classes)
Matrix vectorToOneHotMatrix(const std::vector<int>& indices, int num_classes) {
    int batch_size = indices.size();
    Matrix mat(batch_size, num_classes, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        if (indices[i] >= 0 && indices[i] < num_classes)
            mat.setElement(i, indices[i], 1.0f);
    }
    return mat;
}

void Trainer::train(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches) {
    int num_classes = model.getTargetVocabSize(); 
    
    std::cout << "  Procesando " << source_batches.size() << " muestras..." << std::endl;
    std::cout << "  Vocabulario objetivo: " << num_classes << " clases" << std::endl;
    
    double total_loss = 0.0;
    int processed_samples = 0;
    
    for (size_t i = 0; i < source_batches.size(); ++i) {
        std::cout << "    Muestra " << (i+1) << "/" << source_batches.size() << ": " << std::flush;
        
        // 1. Forward pass
        try {
            Matrix output = model.forward(source_batches[i], target_batches[i]);
            std::cout << "FWD✓ " << std::flush;
            
            // Target simplificado
            int target_length = target_batches[i].size();
            Matrix target(target_length, 1);
            for (int j = 0; j < target_length; ++j) {
                target.setElement(j, 0, target_batches[i][j]);
            }
            
            // 2. Loss con penalización por EOS temprano
            double raw_loss = loss_fn.forward(output, target);
            
            // Aplicar penalización si el modelo predice EOS en posiciones tempranas
            double eos_penalty = calculateEOSPenalty(output, target_batches[i]);
            double final_loss = raw_loss + eos_penalty;
            
            if (eos_penalty > 0.0) {
                std::cout << " [EOS penalty: " << std::fixed << std::setprecision(3) << eos_penalty << "]";
            }
            
            // Calculate adaptive learning rate
            float current_lr = calculateLearningRate(global_step, final_loss);
            global_step++;
            
            std::cout << " Loss: " << std::fixed << std::setprecision(3) << final_loss 
                      << " (LR: " << std::setprecision(4) << current_lr << ")" << std::flush;
            
            // 3. Backward pass - USAR EL NUEVO SISTEMA
            Matrix grad = loss_fn.backward(output, target);
            
            // BACKWARD PASS COMPLETO DEL TRANSFORMER
            model.backward(grad, current_lr);  // Use adaptive learning rate
            
            std::cout << "[UPDATE] Gradientes aplicados con lr=" << std::fixed << std::setprecision(8) << current_lr;
            
            std::cout << " [Updated]" << std::endl;
            
            total_loss += final_loss;
            processed_samples++;
            
        } catch (const std::exception& e) {
            std::cout << " ERROR: " << e.what() << std::endl;
            return;
        }
        
        // Procesar más muestras para mejor aprendizaje
        if (i >= 7) { // Aumentado de 2 a 7 para procesar 8 muestras
            break;
        }
    }
    
    double avg_loss = total_loss / processed_samples;
    std::cout << "  Promedio de loss: " << std::fixed << std::setprecision(3) << avg_loss << std::endl;
}

Trainer::Trainer(Transformer& model, Optimizer& optimizer, Loss& loss_fn, int batch_size, int epochs)
    : model(model), optimizer(optimizer), loss_fn(loss_fn), batch_size(batch_size), epochs(epochs), global_step(0) {
    // Constructor implementation
}

float Trainer::calculateLearningRate(int step, float current_loss) {
    float base_lr = optimizer.getLearningRate();
    
    // Track loss history for better adaptive learning rate
    static std::vector<float> loss_history;
    static int stagnant_count = 0;
    
    loss_history.push_back(current_loss);
    if (loss_history.size() > 15) {
        loss_history.erase(loss_history.begin());
    }
    
    // Warm-up for first 20 steps
    if (step < 20) {
        float warmup_factor = (float)(step + 1) / 20.0f;
        return base_lr * warmup_factor;
    }
    
    // Check if loss is stagnant (last 5 losses)
    if (loss_history.size() >= 8) {
        float recent_avg = 0.0f, older_avg = 0.0f;
        
        // Average of last 4 losses
        for (int i = loss_history.size() - 4; i < loss_history.size(); ++i) {
            recent_avg += loss_history[i];
        }
        recent_avg /= 4.0f;
        
        // Average of 4 losses before that
        for (int i = loss_history.size() - 8; i < loss_history.size() - 4; ++i) {
            older_avg += loss_history[i];
        }
        older_avg /= 4.0f;
        
        float improvement = older_avg - recent_avg;
        
        if (improvement < 0.02f) {  // Very little improvement
            stagnant_count++;
            if (stagnant_count >= 2) {  // Been stagnant for a while
                float adaptive_lr = base_lr * 3.0f;  // Triple the learning rate
                adaptive_lr = std::min(adaptive_lr, 0.05f);  // Cap at 0.05
                
                std::cout << "[TRAINER] Loss stagnant (improvement: " << improvement 
                          << "), boosting LR to " << adaptive_lr << std::endl;
                
                stagnant_count = 0;  // Reset counter
                return adaptive_lr;
            }
        } else {
            stagnant_count = 0;  // Reset if we see improvement
            
            if (improvement > 0.1f) {  // Good improvement
                return base_lr * 0.9f;  // Slightly reduce
            }
        }
    }
    
    return base_lr;
}

double Trainer::calculateEOSPenalty(const Matrix& predictions, const std::vector<int>& target_sequence) {
    const int EOS_TOKEN = 3; // Asumiendo que EOS es el token 3
    double penalty = 0.0;
    
    int batch_size = predictions.getRows();
    int vocab_size = predictions.getCols();
    int expected_length = target_sequence.size();
    
    // Solo penalizar si la secuencia objetivo es suficientemente larga
    if (expected_length <= 2) return 0.0;
    
    for (int pos = 0; pos < std::min(batch_size, expected_length - 1); ++pos) {
        // Obtener la probabilidad softmax del token EOS en esta posición
        float max_logit = predictions.getElement(pos, 0);
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, predictions.getElement(pos, v));
        }
        
        float eos_logit = predictions.getElement(pos, EOS_TOKEN);
        float eos_exp = exp(eos_logit - max_logit);
        
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            sum_exp += exp(predictions.getElement(pos, v) - max_logit);
        }
        
        float eos_prob = eos_exp / sum_exp;
        
        // Calcular penalización basada en posición y probabilidad
        if (pos < expected_length / 2) { // Primera mitad de la secuencia
            float position_penalty = (float)(expected_length / 2 - pos) / (float)(expected_length / 2);
            penalty += position_penalty * eos_prob * 2.0; // Factor de escala
        }
    }
    
    return penalty;
}