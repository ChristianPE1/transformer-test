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
            
            // 2. Loss real
            double loss = loss_fn.forward(output, target);
            std::cout << " Loss: " << std::fixed << std::setprecision(3) << loss << std::flush;
            
            // Calculate adaptive learning rate
            float current_lr = calculateLearningRate(global_step, loss);
            global_step++;
            
            // 3. Backward pass - USAR EL NUEVO SISTEMA
            Matrix grad = loss_fn.backward(output, target);
            
            // BACKWARD PASS COMPLETO DEL TRANSFORMER
            model.backward(grad, current_lr);  // Use adaptive learning rate
            
            std::cout << "[UPDATE] Gradientes aplicados con lr=" << std::fixed << std::setprecision(8) << current_lr;
            
            std::cout << " [Updated]" << std::endl;
            
            total_loss += loss;
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
    : model(model), optimizer(optimizer), loss_fn(loss_fn), batch_size(batch_size), epochs(epochs) {
    // Constructor implementation
}

float Trainer::calculateLearningRate(int step, float current_loss) {
    float base_lr = optimizer.getLearningRate();
    
    // Warm-up for first 50 steps
    if (step < 50) {
        float warmup_factor = (float)(step + 1) / 50.0f; // +1 to avoid lr=0 at step 0
        return base_lr * warmup_factor;
    }
    
    // If loss is stuck (not decreasing), increase learning rate
    if (step > 0 && step % 10 == 0) {
        static float last_loss = current_loss;
        if (current_loss >= last_loss - 0.01f) { // Loss not decreasing significantly
            return base_lr * 1.5f; // Increase by 50%
        }
        last_loss = current_loss;
    }
    
    return base_lr;
}

// Private member to track steps
static int global_step = 0;