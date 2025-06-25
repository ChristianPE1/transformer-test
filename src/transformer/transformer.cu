#include "transformer.cuh"
#include "embeddings.cuh"
#include "../utils/matrix.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

Transformer::Transformer(size_t input_vocab_size, size_t target_vocab_size,size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
    : input_vocab_size(input_vocab_size), target_vocab_size(target_vocab_size),
      d_model(d_model), n_layers(n_layers),
      input_embedding(input_vocab_size, d_model),
      target_embedding(target_vocab_size, d_model),
      pos_encoding(d_model)
{

    std::cout << "Transformer initialized:" << std::endl;
    std::cout << "  Input vocab: " << input_vocab_size << std::endl;
    std::cout << "  Target vocab: " << target_vocab_size << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  layers: " << n_layers << std::endl;
}

Matrix Transformer::encode(const std::vector<int> &input_tokens)
{
    // Get embeddings
    Matrix embeddings = input_embedding.forward(input_tokens);

    // Scale embeddings
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt(d_model);
    for (auto &val : embed_data)
    {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(input_tokens.size());
    Matrix encoder_input = embeddings.add(pos_enc);

    // For now, return encoder_input (no actual encoder layers yet)
    return encoder_input;
}

Matrix Transformer::decode(const std::vector<int> &target_tokens,
                           const Matrix &encoder_output)
{
    // Get target embeddings
    Matrix embeddings = target_embedding.forward(target_tokens);

    // Scale embeddings
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt(d_model);
    for (auto &val : embed_data)
    {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(target_tokens.size());
    Matrix decoder_input = embeddings.add(pos_enc);

    // ATENCIÓN CRUZADA SIMPLE - Mezclar decoder input con encoder output
    Matrix decoder_output = applyCrossAttention(decoder_input, encoder_output);

    return decoder_output;
}

// Nueva función de atención cruzada simple
Matrix Transformer::applyCrossAttention(const Matrix& decoder_input, const Matrix& encoder_output) {
    int decoder_len = decoder_input.getRows();
    int encoder_len = encoder_output.getRows();
    int d_model = decoder_input.getCols();
    
    Matrix attended_output(decoder_len, d_model, 0.0f);
    
    for (int i = 0; i < decoder_len; ++i) {
        for (int d = 0; d < d_model; ++d) {
            float attended_value = 0.0f;
            float attention_sum = 0.0f;
            
            // Calcular atención entre posición i del decoder y todas las del encoder
            for (int j = 0; j < encoder_len; ++j) {
                // Peso de atención simple basado en producto punto
                float attention_score = 0.0f;
                for (int k = 0; k < std::min(16, d_model); ++k) {
                    attention_score += decoder_input.getElement(i, k) * encoder_output.getElement(j, k);
                }
                
                // Normalizar y aplicar softmax simple
                attention_score = exp(attention_score * 0.1f); // Temperature para suavizar
                attention_sum += attention_score;
                
                // Agregar contribución ponderada del encoder
                attended_value += attention_score * encoder_output.getElement(j, d);
            }
            
            // Normalizar por la suma de pesos de atención
            if (attention_sum > 0) {
                attended_value /= attention_sum;
            }
            
            // Combinar con input original (conexión residual)
            float original_value = decoder_input.getElement(i, d);
            float final_value = 0.7f * original_value + 0.3f * attended_value;
            
            attended_output.setElement(i, d, final_value);
        }
    }
    
    return attended_output;
}

Matrix Transformer::forward(const std::vector<int> &source_tokens,
                            const std::vector<int> &target_tokens)
{
    std::cout << "[DEBUG] Forward - source: " << source_tokens.size() 
              << " tokens, target: " << target_tokens.size() << " tokens" << std::endl;
    
    // Store target tokens for later gradient updates
    last_target_tokens = target_tokens;
    
    // Encode
    Matrix encoder_output = encode(source_tokens);
    std::cout << "[DEBUG] Encode OK - shape: " << encoder_output.getRows() << "x" << encoder_output.getCols() << std::endl;

    // Decode
    Matrix decoder_output = decode(target_tokens, encoder_output);
    std::cout << "[DEBUG] Decode OK - shape: " << decoder_output.getRows() << "x" << decoder_output.getCols() << std::endl;    // Project to vocabulary - MEJORADO CON ATENCIÓN A FUENTE
    Matrix output(target_tokens.size(), target_vocab_size, 0.0f);
    std::cout << "[DEBUG] Created output matrix: " << output.getRows() << "x" << output.getCols() << std::endl;

    // Inicializar semilla una vez
    static bool seed_initialized = false;
    if (!seed_initialized) {
        srand(time(nullptr));
        seed_initialized = true;
    }

    for (int i = 0; i < target_tokens.size(); ++i) {
        
        // ATENCIÓN CRUZADA MEJORADA: Diversificar la atención
        std::vector<float> cross_attention(source_tokens.size(), 0.0f);
        float attention_sum = 0.0f;
        
        for (int j = 0; j < source_tokens.size(); ++j) {
            float attention_score = 0.0f;
            
            // Calcular atención basada en:
            // 1. Similitud posicional
            float pos_similarity = 1.0f / (1.0f + abs(i - j));
            
            // 2. Similitud de contenido decoder-encoder 
            for (int d = 0; d < std::min(32, (int)d_model); ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                float encoder_val = encoder_output.getElement(j, d);
                attention_score += decoder_val * encoder_val;
            }
            
            // 3. Componente de diversidad para evitar colapso
            float diversity_factor = 1.0f + 0.3f * sin((i + j) * 0.5f);
            
            // 4. Exploración posicional para que no siempre atienda a pos 0
            float exploration = 0.1f * (j + 1.0f) / source_tokens.size();
            
            attention_score = attention_score * 0.1f + pos_similarity * 0.3f + 
                             diversity_factor * 0.4f + exploration * 0.2f;
            
            cross_attention[j] = exp(attention_score);
            attention_sum += cross_attention[j];
        }
        
        // Normalizar atención
        for (int j = 0; j < source_tokens.size(); ++j) {
            cross_attention[j] /= (attention_sum + 1e-8f);
        }
        
        // DEBUG: Mostrar qué posición atiende más
        int max_att_pos = 0;
        float max_att_val = cross_attention[0];
        for (int j = 1; j < source_tokens.size(); ++j) {
            if (cross_attention[j] > max_att_val) {
                max_att_val = cross_attention[j];
                max_att_pos = j;
            }
        }
        
        if (i % 2 == 0) { // Solo imprimir cada 2 posiciones para no saturar
            std::cout << "[DEBUG] Processed row " << i << " (attending to source pos " << max_att_pos << ")" << std::endl;
        }
        
        for (int v = 0; v < target_vocab_size; ++v) { 
            float similarity = 0.0f;
            
            // 1. Similitud con embedding del token candidato
            std::vector<int> temp_token = {v};
            Matrix vocab_embedding = target_embedding.forward(temp_token);
            
            for (int d = 0; d < std::min(32, (int)d_model); ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                float vocab_val = vocab_embedding.getElement(0, d);
                similarity += decoder_val * vocab_val;
            }
            similarity /= std::min(32, (int)d_model);
            
            // 2. Contribución del contexto fuente usando atención cruzada
            float source_context = 0.0f;
            for (int j = 0; j < source_tokens.size(); ++j) {
                for (int d = 0; d < std::min(8, (int)d_model); ++d) {
                    float encoder_val = encoder_output.getElement(j, d);
                    source_context += encoder_val * cross_attention[j] * ((v + d) % 20 + 1) * 0.01f;
                }
            }
            similarity += source_context;
            
            // 3. Bias de frecuencia ajustado
            if (v < 50) similarity += 0.15f;       // Tokens muy comunes
            else if (v < 200) similarity += 0.1f;  // Tokens comunes  
            else if (v < 500) similarity += 0.05f; // Tokens moderados
            
            // 4. Pequeña exploración aleatoria
            similarity += ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            
            output.setElement(i, v, similarity);
        }
        
        if (i % 2 == 0) {
            // Encontrar posición source con mayor atención
            int max_attention_pos = 0;
            for (int j = 1; j < source_tokens.size(); ++j) {
                if (cross_attention[j] > cross_attention[max_attention_pos]) {
                    max_attention_pos = j;
                }
            }
            std::cout << "[DEBUG] Processed row " << i 
                      << " (attending to source pos " << max_attention_pos << ")" << std::endl;
        }
    }
    
    std::cout << "[DEBUG] Forward completed!" << std::endl;
    return output;
}

std::vector<int> Transformer::generate(const std::vector<int> &source_tokens,
int sos_token, int eos_token, size_t max_length)
{
    std::vector<int> generated = {sos_token};
    
    // Estimar longitud objetivo basada en la longitud de entrada
    size_t target_length = std::max(2, (int)(source_tokens.size() * 0.9)); // 90% de la entrada
    size_t actual_max = std::min(max_length, target_length + 2); // +2 para flexibilidad

    for (size_t step = 0; step < actual_max; ++step)
    {
        Matrix output = forward(source_tokens, generated);

        // Get last token predictions
        int last_pos = generated.size() - 1;
        
        // Buscar el mejor token con filtros
        std::vector<std::pair<float, int>> candidates;
        int search_limit = std::min(1000, (int)target_vocab_size);
        
        for (int v = 0; v < search_limit; ++v)
        {
            float score = output.getElement(last_pos, v);
            
            // FILTROS IMPORTANTES:
            // 1. No repetir SOS después del primer token
            if (v == sos_token && generated.size() > 1) {
                score -= 10.0f; // Penaliza fuertemente
            }
            
            // 2. Si ya llevamos suficientes tokens, priorizar EOS
            if (generated.size() >= target_length && v == eos_token) {
                score += 5.0f; // Boost fuerte para EOS cuando debería terminar
            }
            
            // 3. Penalizar tokens muy recientes (evitar repeticiones)
            for (int i = std::max(0, (int)generated.size() - 3); i < generated.size(); i++) {
                if (generated[i] == v) {
                    score -= 2.0f; // Penaliza repeticiones
                    break;
                }
            }
            
            candidates.push_back({score, v});
        }
        
        // Ordenar por score descendente
        std::sort(candidates.begin(), candidates.end(), std::greater<std::pair<float, int>>());
        
        // Selección inteligente del token
        int best_token = candidates[0].second;
        float best_score = candidates[0].first;
        
        // Usar sampling solo en los primeros tokens para más variedad
        if (step < 2 && candidates.size() > 5) {
            float temperature = 1.2f;
            std::vector<float> probs(5);
            float sum = 0.0f;
            
            for (int i = 0; i < 5; ++i) {
                probs[i] = exp(candidates[i].first / temperature);
                sum += probs[i];
            }
            
            // Normalizar probabilidades
            for (int i = 0; i < 5; ++i) {
                probs[i] /= sum;
            }
            
            // Selección probabilística entre top 3
            float rand_val = ((float)rand() / RAND_MAX);
            float cumsum = 0.0f;
            for (int i = 0; i < 3; ++i) {
                cumsum += probs[i];
                if (rand_val <= cumsum) {
                    best_token = candidates[i].second;
                    best_score = candidates[i].first;
                    break;
                }
            }
        }

        // DEBUG: Muestra información de generación
        if (step < 3) {
            std::cout << "[GEN] Step " << step << " - Best token: " << best_token 
                      << " (score: " << std::fixed << std::setprecision(1) << best_score 
                      << ", target_len: " << target_length << ")";
            
            std::cout << " [Top scores: ";
            for (int i = 0; i < std::min(5, (int)candidates.size()); ++i) {
                std::cout << candidates[i].second << ":" << std::fixed << std::setprecision(1) << candidates[i].first << " ";
            }
            std::cout << "]" << std::endl;
        }

        generated.push_back(best_token);

        // Terminar si encontramos EOS
        if (best_token == eos_token) {
            break;
        }
        
        // Forzar terminación si se excede la longitud objetivo
        if (generated.size() >= target_length + 1) {
            generated.push_back(eos_token);
            break;
        }
    }
    
    // Asegurar que termine con EOS si no lo tiene
    if (generated.back() != eos_token && generated.size() < max_length) {
        generated.push_back(eos_token);
    }

    return generated;
}

void Transformer::updateWeights(const Matrix& gradients, float learning_rate) {
    std::cout << "[UPDATE] Aplicando gradientes con lr=" << learning_rate << std::endl;
    
    // Verificar que el learning rate no sea cero
    if (learning_rate == 0.0f) {
        std::cout << "[UPDATE] WARNING: Learning rate es 0! Los pesos no se actualizarán." << std::endl;
        return;
    }
    
    // Usar los tokens del último forward pass
    if (!last_target_tokens.empty()) {
        try {
            // Verificar dimensiones de gradientes
            std::cout << "[UPDATE] Gradientes: " << gradients.getRows() << "x" << gradients.getCols() << std::endl;
            std::cout << "[UPDATE] Tokens objetivo: " << last_target_tokens.size() << std::endl;
            
            target_embedding.updateWeights(gradients, learning_rate, last_target_tokens);
            std::cout << "[UPDATE] Target embeddings actualizados exitosamente para " << last_target_tokens.size() << " tokens" << std::endl;
            
            // Log algunos valores de ejemplo para debug
            std::vector<float> sample_grads;
            gradients.copyToHost(sample_grads);
            if (!sample_grads.empty()) {
                std::cout << "[UPDATE] Muestra de gradientes: ";
                for (int i = 0; i < std::min(5, (int)sample_grads.size()); ++i) {
                    std::cout << std::fixed << std::setprecision(4) << sample_grads[i] << " ";
                }
                std::cout << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "[UPDATE] Error actualizando embeddings: " << e.what() << std::endl;
        }
    } else {
        std::cout << "[UPDATE] No hay tokens para actualizar" << std::endl;
    }
}

// Add backward pass method
void Transformer::backward(const Matrix& grad_output, float learning_rate) {
    // Simplified backward pass - focus on updating the embeddings
    
    std::cout << "[BACKWARD] Starting backward pass..." << std::endl;
    
    // For now, we'll focus on updating the target embeddings
    // In a full implementation, you'd also update encoder/decoder layers
    
    // Update target embeddings using the gradient
    updateTargetEmbeddings(grad_output, learning_rate);
    
    std::cout << "[BACKWARD] Completed backward pass with lr=" << learning_rate << std::endl;
}

void Transformer::updateTargetEmbeddings(const Matrix& gradients, float learning_rate) {
    // Update target embeddings based on gradients and last used tokens
    
    if (last_target_tokens.empty()) {
        std::cout << "[WARNING] No target tokens stored for gradient update" << std::endl;
        return;
    }
    
    // Use the existing updateWeights method of the embedding class
    target_embedding.updateWeights(gradients, learning_rate, last_target_tokens);
    
    std::cout << "[UPDATE] Target embeddings updated for " << last_target_tokens.size() << " tokens" << std::endl;
}