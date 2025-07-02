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

int main()
{
    try
    {
        std::cout << "=== CUDA Transformer with TSV Dataset ===" << std::endl;

        // Verify CUDA
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cout << "CUDA devices found: " << deviceCount << std::endl;

        if (deviceCount == 0)
        {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        // Load and process dataset
        Dataset dataset;
        std::cout << "Loading TSV file..." << std::endl;
        dataset.loadTSV("db_translate.tsv");

        std::cout << "Building vocabularies..." << std::endl;
        dataset.buildVocabularies();

        std::cout << "Creating train/test split..." << std::endl;
        dataset.createTrainTestSplit(0.8f);

        std::cout << "\nDataset Statistics:" << std::endl;
        std::cout << "English vocab size: " << dataset.getEngVocab().size() << std::endl;
        std::cout << "Spanish vocab size: " << dataset.getSpaVocab().size() << std::endl;
        std::cout << "Training samples: " << dataset.getTrainSize() << std::endl;
        std::cout << "Test samples: " << dataset.getTestSize() << std::endl;

        // Test vocabulary
        std::cout << "\n=== Vocabulary Test ===" << std::endl;
        const auto &eng_vocab = dataset.getEngVocab();
        const auto &spa_vocab = dataset.getSpaVocab();

        // Test English sentence
        std::string test_eng = "<sos> hello world <eos>";
        auto eng_ids = eng_vocab.sentenceToIds(test_eng);
        std::cout << "English: \"" << test_eng << "\" -> ";
        for (int id : eng_ids)
        {
            std::cout << id << " ";
        }
        std::cout << "-> \"" << eng_vocab.idsToSentence(eng_ids) << "\"" << std::endl;

        // Test Spanish sentence
        std::string test_spa = "<sos> hola mundo <eos>";
        auto spa_ids = spa_vocab.sentenceToIds(test_spa);
        std::cout << "Spanish: \"" << test_spa << "\" -> ";
        for (int id : spa_ids)
        {
            std::cout << id << " ";
        }
        std::cout << "-> \"" << spa_vocab.idsToSentence(spa_ids) << "\"" << std::endl;

        // Test batch loading
        std::cout << "\n=== Batch Test ===" << std::endl;
        auto batch = dataset.getBatch(3, true);
        std::cout << "Loaded batch of " << batch.size() << " samples:" << std::endl;

        for (size_t i = 0; i < batch.size(); ++i)
        {
            const auto &sample = batch[i];
            const auto &eng_ids = sample.first;
            const auto &spa_ids = sample.second;
            std::cout << "Sample " << i << ":" << std::endl;
            std::cout << "  ENG: " << eng_vocab.idsToSentence(eng_ids) << std::endl;
            std::cout << "  SPA: " << spa_vocab.idsToSentence(spa_ids) << std::endl;
        }

        std::cout << "\n=== Success! Dataset ready for training ===" << std::endl;

        // Test Transformer
        std::cout << "\n=== Testing Transformer ===" << std::endl;
        Transformer transformer(dataset.getEngVocab().size(),dataset.getSpaVocab().size(),
                                128,  // d_model
                                4,    // n_heads
                                2,    // n_layers
                                256); // d_ff

        // Test forward pass
        auto test_batch = dataset.getBatch(1, true);
        if (!test_batch.empty())
        {
            const auto &sample = test_batch[0];
            const auto &source_ids = sample.first;
            const auto &target_ids = sample.second;

            std::cout << "Testing forward pass with:" << std::endl;
            std::cout << "  Source: " << eng_vocab.idsToSentence(source_ids) << std::endl;
            std::cout << "  Target: " << spa_vocab.idsToSentence(target_ids) << std::endl;

            Matrix output = transformer.forward(source_ids, target_ids);
            std::cout << "Forward pass completed!" << std::endl;
            std::cout << "Output shape: " << output.getRows() << "x" << output.getCols() << std::endl;

            // Test generation
            std::cout << "\nTesting generation..." << std::endl;
            auto generated = transformer.generate(source_ids, 2, 3, 10); // sos=2, eos=3
            std::cout << "Generated: " << spa_vocab.idsToSentence(generated) << std::endl;
            
            // AGREGAR ENTRENAMIENTO AQUÍ:
            std::cout << "\n=== Iniciando Entrenamiento Optimizado ===" << std::endl;
            
            // Configuración de entrenamiento optimizada
            int epochs = 150;  // Más épocas para llegar a pérdida ~3.0
            int batch_size = 16;   // Batch más grande para mejor eficiencia  
            float base_learning_rate = 0.005f;  // Aumentar LR inicial para acelerar
            const float max_learning_rate = 0.02f;  // Aumentar límite máximo
            const float min_learning_rate = 0.001f; // Aumentar límite mínimo
            
            std::cout << "Configuración:" << std::endl;
            std::cout << "  Épocas: " << epochs << std::endl;
            // Configuración de entrenamiento simplificada
            std::cout << "  Batch size: " << batch_size << std::endl;
            std::cout << "  Learning rate: " << base_learning_rate << std::endl;
            
            // Crear archivo para guardar la pérdida
            std::ofstream loss_file("training_loss.txt");
            if (!loss_file.is_open()) {
                std::cerr << "Error: No se pudo crear el archivo training_loss.txt" << std::endl;
                return 1;
            }
            
            // Escribir encabezado
            loss_file << "Epoch,Loss,CurrentLR,BestLoss,StagnantEpochs" << std::endl;
            std::cout << "Archivo training_loss.txt creado para guardar la pérdida por época." << std::endl;
            
            // Crear componentes de entrenamiento (SIMPLE)
            CrossEntropyLoss loss_fn;
            SGD optimizer(base_learning_rate, 0.0f);  // Sin momentum - más simple
            Trainer trainer(transformer, optimizer, loss_fn, batch_size, epochs);
            
            // Bucle de entrenamiento SIMPLE
            float best_loss = std::numeric_limits<float>::max();
            float initial_loss = 0.0f;
            int stagnant_epochs = 0;
            
            for (int epoch = 0; epoch < epochs; epoch++) {
                std::cout << "\nÉpoca " << (epoch + 1) << "/" << epochs << std::endl;
                
                // Obtener batch de entrenamiento
                auto train_batch = dataset.getBatch(batch_size, true);
                
                std::vector<std::vector<int>> source_batches;
                std::vector<std::vector<int>> target_batches;
                
                for (const auto& sample : train_batch) {
                    source_batches.push_back(sample.first);
                    target_batches.push_back(sample.second);
                }
                
                // Entrenar y calcular pérdida manualmente
                double total_loss = 0.0;
                int samples_processed = 0;
                
                // Learning rate con actualización correcta
                float current_lr = base_learning_rate;
                
                // Actualizar el learning rate en el optimizador
                optimizer.setLearningRate(current_lr);
                
                // Procesar el batch y calcular pérdida promedio
                for (size_t i = 0; i < std::min(source_batches.size(), static_cast<size_t>(8)); ++i) {
                    try {
                        // Forward pass
                        Matrix output = transformer.forward(source_batches[i], target_batches[i]);
                        
                        // Crear target matrix
                        int target_length = target_batches[i].size();
                        Matrix target(target_length, 1);
                        for (int j = 0; j < target_length; ++j) {
                            target.setElement(j, 0, target_batches[i][j]);
                        }
                        
                        // Calcular pérdida
                        double sample_loss = loss_fn.forward(output, target);
                        total_loss += sample_loss;
                        samples_processed++;
                        
                        // Backward pass con el learning rate actual
                        Matrix grad = loss_fn.backward(output, target);
                        transformer.backward(grad, current_lr);
                        
                    } catch (const std::exception& e) {
                        std::cout << "Error en muestra " << i << ": " << e.what() << std::endl;
                        continue;
                    }
                }
                
                float epoch_loss = (samples_processed > 0) ? (total_loss / samples_processed) : 0.0f;
                
                // VERIFICAR NaN/Inf en el loss - detener entrenamiento si ocurre
                if (std::isnan(epoch_loss) || std::isinf(epoch_loss)) {
                    std::cout << "❌ ERROR: Loss inválido detectado (NaN/Inf). Deteniendo entrenamiento." << std::endl;
                    std::cout << "Esto indica inestabilidad numérica. Considera reducir el learning rate." << std::endl;
                    break;
                }
                
                // NUEVO SISTEMA DE ADAPTACIÓN DE LEARNING RATE CON LÍMITES - MÁS AGRESIVO
                if (epoch > 10 && epoch_loss < best_loss * 0.995f) {
                    // Si mejoramos >0.5%, acelerar más agresivamente (máx 5%)
                    base_learning_rate = std::min(base_learning_rate * 1.05f, max_learning_rate);
                } else if (epoch > 15 && stagnant_epochs > 5) {
                    // Si estamos estancados >5 épocas, reducir LR levemente
                    base_learning_rate = std::max(base_learning_rate * 0.98f, min_learning_rate);
                }
                
                // Limitar el learning rate por seguridad
                base_learning_rate = std::max(min_learning_rate, std::min(max_learning_rate, base_learning_rate));
                
                // Guardar estadísticas
                if (epoch == 0) initial_loss = epoch_loss;
                if (epoch_loss < best_loss) {
                    best_loss = epoch_loss;
                    stagnant_epochs = 0;
                } else {
                    stagnant_epochs++;
                }
                
                // Guardar pérdida en archivo con el learning rate CORRECTO
                loss_file << (epoch + 1) << "," << std::fixed << std::setprecision(6) << epoch_loss 
                         << "," << base_learning_rate << "," << best_loss << "," << stagnant_epochs << std::endl;
                loss_file.flush(); // Asegurar que se escriba inmediatamente
                
                // Progress report CADA ÉPOCA para ver claramente el progreso del loss
                std::cout << "Epoca " << (epoch + 1) << "/" << epochs << " - Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                         << " | LR: " << std::scientific << std::setprecision(3) << base_learning_rate;
                if (epoch_loss < best_loss) std::cout << " [MEJOR]";
                if (stagnant_epochs > 5) std::cout << " [Estancado:" << stagnant_epochs << "]";
                
                // Test generation cada 10 épocas para verificar traducción
                if ((epoch + 1) % 10 == 0) {
                    auto gen = transformer.generate(source_ids, 2, 3, 5); // Menos tokens
                    std::cout << " | Test: " << spa_vocab.idsToSentence(gen);
                }
                std::cout << std::endl;
                
                // Advertencia si está estancado
                if (stagnant_epochs > 20) {
                    std::cout << "⚠️  Pérdida estancada por " << stagnant_epochs << " épocas" << std::endl;
                }
            }
            
            // Cerrar archivo
            loss_file.close();
            
            // Resumen de entrenamiento
            float improvement = initial_loss - best_loss;
            float improvement_percent = (improvement / initial_loss) * 100.0f;
            
            std::cout << "\n=== RESUMEN DE ENTRENAMIENTO ===" << std::endl;
            std::cout << "Pérdida inicial: " << std::fixed << std::setprecision(4) << initial_loss << std::endl;
            std::cout << "Mejor pérdida: " << best_loss << std::endl;
            std::cout << "Mejora absoluta: " << improvement << std::endl;
            std::cout << "Mejora porcentual: " << improvement_percent << "%" << std::endl;
            std::cout << "¡Entrenamiento completado! Pérdida guardada en training_loss.txt" << std::endl;
        }

        // PRUEBAS DE TRADUCCIÓN
        std::cout << "\n=== Pruebas de Traducción ===" << std::endl;

        std::vector<std::string> test_sentences = {
            "<sos> hello <eos>",
            "<sos> how are you <eos>",
            "<sos> good morning <eos>",
            "<sos> thank you <eos>",
            "<sos> i love you <eos>"
        };

        for (const auto& sentence : test_sentences) {
            auto test_ids = eng_vocab.sentenceToIds(sentence);
            auto generated = transformer.generate(test_ids, 2, 3, 10);
            
            std::cout << "ENG: " << sentence << std::endl;
            std::cout << "ESP: " << spa_vocab.idsToSentence(generated) << std::endl;
            std::cout << "---" << std::endl;
        }

        // CARGA Y PROCESAMIENTO DEL DATASET MNIST
        std::cout << "\n=== Cargando y Procesando Dataset MNIST ===" << std::endl;

        // Rutas de los archivos de imágenes y etiquetas de entrenamiento y prueba
        std::string train_images_path = "data/train-images.idx3-ubyte";
        std::string train_labels_path = "data/train-labels.idx1-ubyte";
        std::string test_images_path = "data/t10k-images.idx3-ubyte";
        std::string test_labels_path = "data/t10k-labels.idx1-ubyte";

        // Cargar dataset MNIST
        MNISTLoader mnist_loader;
        auto train_data = mnist_loader.load(train_images_path, train_labels_path);
        auto test_data = mnist_loader.load(test_images_path, test_labels_path);

        // Procesar datos MNIST en parches
        int patch_size = 4; // Tamaño de parche de ejemplo
        auto train_patches = mnist_loader.create_patches(train_data.images, patch_size);
        auto test_patches = mnist_loader.create_patches(test_data.images, patch_size);

        // Inicializar modelo Vision Transformer
        VisionTransformer vit_model(128, 4, 2, 256);
        vit_model.initialize(patch_size, 10); // 10 clases para MNIST

        // Bucle de entrenamiento para MNIST
        int num_epochs = 5; // Número de épocas para entrenamiento MNIST
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = 0.0f;
            for (auto& batch : train_patches) {
                auto predictions = vit_model.forward(batch);
                auto loss = compute_loss(predictions, batch.labels);
                vit_model.backward(loss);
                vit_model.update_weights();
                epoch_loss += loss;
            }
            std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
        }

        std::cout << "¡Entrenamiento de MNIST completado!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}