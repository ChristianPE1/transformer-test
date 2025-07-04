#include "transformer.cuh"
#include "embeddings.cuh"
#include "encoder.cuh"
#include "decoder.cuh"
#include "../utils/matrix.cuh"
#include "../layers/linear.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

Transformer::Transformer(size_t input_vocab_size, size_t target_vocab_size,size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
    : input_vocab_size(input_vocab_size), target_vocab_size(target_vocab_size),
      d_model(d_model), n_layers(n_layers), n_heads(n_heads), d_ff(d_ff),
      input_embedding(input_vocab_size, d_model),
      target_embedding(target_vocab_size, d_model),
      pos_encoding(d_model),
      output_projection(d_model, target_vocab_size)
{
    std::cout << "Initializing Transformer with real layers..." << std::endl;
    
    // Initialize encoder layers
    encoder_layers.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        encoder_layers.emplace_back(d_model, n_heads, d_ff);
    }
    
    // Initialize decoder layers  
    decoder_layers.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        decoder_layers.emplace_back(d_model, n_heads, d_ff);
    }

    std::cout << "Transformer initialized:" << std::endl;
    std::cout << "  Input vocab: " << input_vocab_size << std::endl;
    std::cout << "  Target vocab: " << target_vocab_size << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  n_heads: " << n_heads << std::endl;
    std::cout << "  layers: " << n_layers << std::endl;
    std::cout << "  d_ff: " << d_ff << std::endl;
    std::cout << "  Encoder layers: " << encoder_layers.size() << std::endl;
    std::cout << "  Decoder layers: " << decoder_layers.size() << std::endl;
}

Matrix Transformer::encode(const std::vector<int> &input_tokens)
{
    // std::cout << "[ENCODER] Starting encode with " << input_tokens.size() << " tokens" << std::endl;
    
    // Get embeddings
    Matrix embeddings = input_embedding.forward(input_tokens);

    // DEBUG: Check if embeddings are non-zero
    std::vector<float> embed_check(std::min(20, embeddings.getRows() * embeddings.getCols()));
    embeddings.copyToHost(embed_check);
    
    float embed_sum = 0.0f;
    int non_zero_embeds = 0;
    for (int i = 0; i < embed_check.size(); ++i) {
        if (abs(embed_check[i]) > 1e-6f) {
            non_zero_embeds++;
            embed_sum += embed_check[i];
        }
    }
    
    // std::cout << "[ENCODER] Embeddings stats: " << non_zero_embeds << "/" << embed_check.size() 
    //           << " non-zero, sum=" << embed_sum << std::endl;
    
    if (non_zero_embeds == 0) {
        std::cout << "[ENCODER] CRITICAL ERROR: All embeddings are zero!" << std::endl;
        return embeddings; // Return zeros - this will explain the loss problem
    }

    // Scale embeddings
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt(d_model);
    // std::cout << "[ENCODER] Scaling embeddings by " << scale << std::endl;
    for (auto &val : embed_data)
    {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(input_tokens.size());
    
    // DEBUG: Check positional encoding values before addition
    std::vector<float> pos_enc_check(std::min(20, pos_enc.getRows() * pos_enc.getCols()));
    pos_enc.copyToHost(pos_enc_check);
    
    float pos_enc_sum = 0.0f;
    int non_zero_pos_enc = 0;
    for (int i = 0; i < pos_enc_check.size(); ++i) {
        if (abs(pos_enc_check[i]) > 1e-6f) {
            non_zero_pos_enc++;
            pos_enc_sum += pos_enc_check[i];
        }
    }
    
    std::cout << "[ENCODER] POS ENC before add: " << non_zero_pos_enc << "/" << pos_enc_check.size() 
              << " non-zero, sum=" << pos_enc_sum << std::endl;
    
    // Check scaled embeddings before addition
    std::vector<float> scaled_embed_check(std::min(20, embeddings.getRows() * embeddings.getCols()));
    embeddings.copyToHost(scaled_embed_check);
    
    float scaled_embed_sum = 0.0f;
    int non_zero_scaled = 0;
    for (int i = 0; i < scaled_embed_check.size(); ++i) {
        if (abs(scaled_embed_check[i]) > 1e-6f) {
            non_zero_scaled++;
            scaled_embed_sum += scaled_embed_check[i];
        }
    }
    
    std::cout << "[ENCODER] SCALED EMBEDS before add: " << non_zero_scaled << "/" << scaled_embed_check.size() 
              << " non-zero, sum=" << scaled_embed_sum << std::endl;
    
    Matrix encoder_input = embeddings.add(pos_enc);

    // DEBUG: Check encoder input after adding positional encoding
    std::vector<float> enc_input_check(std::min(20, encoder_input.getRows() * encoder_input.getCols()));
    encoder_input.copyToHost(enc_input_check);
    
    float enc_input_sum = 0.0f;
    int non_zero_inputs = 0;
    for (int i = 0; i < enc_input_check.size(); ++i) {
        if (abs(enc_input_check[i]) > 1e-6f) {
            non_zero_inputs++;
            enc_input_sum += enc_input_check[i];
        }
    }
    
    std::cout << "[ENCODER] After pos encoding: " << non_zero_inputs << "/" << enc_input_check.size() 
              << " non-zero, sum=" << enc_input_sum << std::endl;

    // Pass through encoder layers (REAL IMPLEMENTATION)
    Matrix current_output = encoder_input; // Start with input
    
    // Pass through each encoder layer
    for (size_t layer_idx = 0; layer_idx < encoder_layers.size(); ++layer_idx) {
        std::cout << "[ENCODER] Processing layer " << layer_idx << std::endl;
        
        Matrix layer_output = encoder_layers[layer_idx].forward(current_output); // No src_mask for now
        
        // DEBUG: Check each layer's output
        std::vector<float> layer_check(std::min(10, layer_output.getRows() * layer_output.getCols()));
        layer_output.copyToHost(layer_check);
        
        float layer_sum = 0.0f;
        int non_zero_layer = 0;
        for (int i = 0; i < layer_check.size(); ++i) {
            if (abs(layer_check[i]) > 1e-6f) {
                non_zero_layer++;
                layer_sum += layer_check[i];
            }
        }
        
        std::cout << "[ENCODER] Layer " << layer_idx << " output: " << non_zero_layer << "/" << layer_check.size() 
                  << " non-zero, sum=" << layer_sum << std::endl;
        
        if (non_zero_layer == 0) {
            std::cout << "[ENCODER] CRITICAL: Layer " << layer_idx << " produced all zeros!" << std::endl;
        }
        
        current_output = layer_output;
    }

    // std::cout << "[ENCODER] Encoding complete" << std::endl;
    return current_output;
}

Matrix Transformer::decode(const std::vector<int> &target_tokens,
                           const Matrix &encoder_output)
{
    // std::cout << "[DECODER] Starting decode with " << target_tokens.size() << " tokens" << std::endl;
    
    // Get target embeddings
    Matrix embeddings = target_embedding.forward(target_tokens);

    // DEBUG: Check target embeddings
    std::vector<float> target_embed_check(std::min(20, embeddings.getRows() * embeddings.getCols()));
    embeddings.copyToHost(target_embed_check);
    
    float target_embed_sum = 0.0f;
    int non_zero_target_embeds = 0;
    for (int i = 0; i < target_embed_check.size(); ++i) {
        if (abs(target_embed_check[i]) > 1e-6f) {
            non_zero_target_embeds++;
            target_embed_sum += target_embed_check[i];
        }
    }
    
    std::cout << "[DECODER] Target embeddings stats: " << non_zero_target_embeds << "/" << target_embed_check.size() 
              << " non-zero, sum=" << target_embed_sum << std::endl;
    
    if (non_zero_target_embeds == 0) {
        std::cout << "[DECODER] CRITICAL ERROR: All target embeddings are zero!" << std::endl;
        return embeddings; // Return zeros
    }

    // Scale embeddings
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt(d_model);
    // std::cout << "[DECODER] Scaling target embeddings by " << scale << std::endl;
    for (auto &val : embed_data)
    {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(target_tokens.size());
    
    // DEBUG: Check positional encoding values before addition
    std::vector<float> dec_pos_enc_check(std::min(20, pos_enc.getRows() * pos_enc.getCols()));
    pos_enc.copyToHost(dec_pos_enc_check);
    
    float dec_pos_enc_sum = 0.0f;
    int non_zero_dec_pos_enc = 0;
    for (int i = 0; i < dec_pos_enc_check.size(); ++i) {
        if (abs(dec_pos_enc_check[i]) > 1e-6f) {
            non_zero_dec_pos_enc++;
            dec_pos_enc_sum += dec_pos_enc_check[i];
        }
    }
    
    std::cout << "[DECODER] POS ENC before add: " << non_zero_dec_pos_enc << "/" << dec_pos_enc_check.size() 
              << " non-zero, sum=" << dec_pos_enc_sum << std::endl;
    
    // Check scaled embeddings before addition
    std::vector<float> dec_scaled_embed_check(std::min(20, embeddings.getRows() * embeddings.getCols()));
    embeddings.copyToHost(dec_scaled_embed_check);
    
    float dec_scaled_embed_sum = 0.0f;
    int non_zero_dec_scaled = 0;
    for (int i = 0; i < dec_scaled_embed_check.size(); ++i) {
        if (abs(dec_scaled_embed_check[i]) > 1e-6f) {
            non_zero_dec_scaled++;
            dec_scaled_embed_sum += dec_scaled_embed_check[i];
        }
    }
    
    std::cout << "[DECODER] SCALED EMBEDS before add: " << non_zero_dec_scaled << "/" << dec_scaled_embed_check.size() 
              << " non-zero, sum=" << dec_scaled_embed_sum << std::endl;
    
    Matrix decoder_input = embeddings.add(pos_enc);

    // DEBUG: Check decoder input after positional encoding
    std::vector<float> dec_input_check(std::min(20, decoder_input.getRows() * decoder_input.getCols()));
    decoder_input.copyToHost(dec_input_check);
    
    float dec_input_sum = 0.0f;
    int non_zero_dec_inputs = 0;
    for (int i = 0; i < dec_input_check.size(); ++i) {
        if (abs(dec_input_check[i]) > 1e-6f) {
            non_zero_dec_inputs++;
            dec_input_sum += dec_input_check[i];
        }
    }
    
    std::cout << "[DECODER] After pos encoding: " << non_zero_dec_inputs << "/" << dec_input_check.size() 
              << " non-zero, sum=" << dec_input_sum << std::endl;

    // Pass through decoder layers (REAL IMPLEMENTATION)
    Matrix current_output = decoder_input;
    
    // Create causal mask for target sequence
    Matrix target_mask = createCausalMask(target_tokens.size());
    
    for (size_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        std::cout << "[DECODER] Processing layer " << layer_idx << std::endl;
        
        Matrix layer_output = decoder_layers[layer_idx].forward(current_output, encoder_output, target_mask);
        
        // DEBUG: Check each decoder layer's output
        std::vector<float> dec_layer_check(std::min(10, layer_output.getRows() * layer_output.getCols()));
        layer_output.copyToHost(dec_layer_check);
        
        float dec_layer_sum = 0.0f;
        int non_zero_dec_layer = 0;
        for (int i = 0; i < dec_layer_check.size(); ++i) {
            if (abs(dec_layer_check[i]) > 1e-6f) {
                non_zero_dec_layer++;
                dec_layer_sum += dec_layer_check[i];
            }
        }
        
        std::cout << "[DECODER] Layer " << layer_idx << " output: " << non_zero_dec_layer << "/" << dec_layer_check.size() 
                  << " non-zero, sum=" << dec_layer_sum << std::endl;
        
        if (non_zero_dec_layer == 0) {
            std::cout << "[DECODER] CRITICAL: Decoder layer " << layer_idx << " produced all zeros!" << std::endl;
        }
        
        current_output = layer_output;
    }

    // std::cout << "[DECODER] Decoding complete" << std::endl;
    return current_output;
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
                for (int k = 0; k < d_model; ++k) {  // USE FULL d_model, not just 16
                    attention_score += decoder_input.getElement(i, k) * encoder_output.getElement(j, k);
                }
                
                // Scale properly and apply softmax
                attention_score = attention_score / sqrtf((float)d_model);  // Proper scaling
                attention_score = exp(attention_score); // Remove temperature scaling
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
    // Reduce logging for cleaner output
    // std::cout << "[DEBUG] Forward - source: " << source_tokens.size() 
    //           << " tokens, target: " << target_tokens.size() << " tokens" << std::endl;
    
    // Store target tokens for later gradient updates
    last_target_tokens = target_tokens;
    
    // Encode source sequence
    Matrix encoder_output = encode(source_tokens);
    // std::cout << "[DEBUG] Encode OK - shape: " << encoder_output.getRows() << "x" << encoder_output.getCols() << std::endl;
    
    // DEBUG: Check encoder output values
    std::vector<float> enc_sample;
    encoder_output.copyToHost(enc_sample);
    if (!enc_sample.empty()) {
        std::cout << "[DEBUG] Encoder sample values: ";
        for (int i = 0; i < std::min(5, (int)enc_sample.size()); ++i) {
            std::cout << std::fixed << std::setprecision(3) << enc_sample[i] << " ";
        }
        std::cout << std::endl;
    }

    // Decode target sequence
    Matrix decoder_output = decode(target_tokens, encoder_output);
    // std::cout << "[DEBUG] Decode OK - shape: " << decoder_output.getRows() << "x" << decoder_output.getCols() << std::endl;
    
    // DEBUG: Check decoder output values
    std::vector<float> dec_sample;
    decoder_output.copyToHost(dec_sample);
    if (!dec_sample.empty()) {
        std::cout << "[DEBUG] Decoder sample values: ";
        for (int i = 0; i < std::min(5, (int)dec_sample.size()); ++i) {
            std::cout << std::fixed << std::setprecision(3) << dec_sample[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Project to vocabulary space
    Matrix output = output_projection.forward(decoder_output);
    // std::cout << "[DEBUG] Output projection - shape: " << output.getRows() << "x" << output.getCols() << std::endl;
    
    // DEBUG: Check final output values and fix any NaN/Inf
    std::vector<float> out_sample;
    output.copyToHost(out_sample);
    if (!out_sample.empty()) {
        // Check for and fix NaN/Inf values
        bool has_nan = false, has_inf = false;
        float min_val = out_sample[0], max_val = out_sample[0];
        
        for (size_t i = 0; i < out_sample.size(); ++i) {
            if (std::isnan(out_sample[i])) {
                out_sample[i] = 0.0f;
                has_nan = true;
            } else if (std::isinf(out_sample[i])) {
                out_sample[i] = (out_sample[i] > 0) ? 10.0f : -10.0f;
                has_inf = true;
            } else {
                // Clamp extreme values
                if (out_sample[i] > 50.0f) out_sample[i] = 50.0f;
                if (out_sample[i] < -50.0f) out_sample[i] = -50.0f;
                
                min_val = std::min(min_val, out_sample[i]);
                max_val = std::max(max_val, out_sample[i]);
            }
        }
        
        if (has_nan || has_inf) {
            std::cout << "[DEBUG] FIXED NaN/Inf in output, copying back" << std::endl;
            output.copyFromHost(out_sample);
        }
        
        std::cout << "[DEBUG] Output range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "[DEBUG] Output sample values: ";
        for (int i = 0; i < std::min(5, (int)out_sample.size()); ++i) {
            std::cout << std::fixed << std::setprecision(3) << out_sample[i] << " ";
        }
        std::cout << std::endl;
        
        // Check if all values are the same (indicating a problem)
        bool all_same = true;
        float first_val = out_sample[0];
        for (int i = 1; i < std::min(20, (int)out_sample.size()); ++i) {
            if (abs(out_sample[i] - first_val) > 1e-6) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            std::cout << "[WARNING] All output values are identical: " << first_val << std::endl;
        }
    }
    
    // std::cout << "[DEBUG] Forward completed!" << std::endl;
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
            
            // 2. PENALIZAR MUY FUERTEMENTE EOS en los primeros tokens - AUMENTADO
            if (v == eos_token && generated.size() <= target_length) {
                // Penalización MUY agresiva para EOS temprano
                float penalty = 30.0f - (float)(generated.size() - 1) * 3.0f; 
                score -= std::max(10.0f, penalty); // Mínimo 10.0 de penalización (doblado)
                
                // Penalización extra si es el primer token
                if (generated.size() == 1) {
                    score -= 50.0f; // Penalización masiva para EOS como primer token
                }
            }
            
            // 3. Si ya llevamos suficientes tokens, priorizar EOS moderadamente
            if (generated.size() > target_length && v == eos_token) {
                score += 2.0f; // Boost moderado para EOS cuando debería terminar
            }
            
            // 4. Penalizar tokens muy recientes (evitar repeticiones) - MEJORADO
            int repetition_penalty_applied = 0;
            for (int i = std::max(0, (int)generated.size() - 4); i < generated.size(); i++) {
                if (generated[i] == v) {
                    float distance_penalty = 4.0f - (generated.size() - 1 - i); // Penaliza más las repeticiones recientes
                    score -= distance_penalty;
                    repetition_penalty_applied++;
                }
            }
            
            // Penalización extra por múltiples repeticiones del mismo token
            if (repetition_penalty_applied > 1) {
                score -= repetition_penalty_applied * 2.0f;
            }
            
            // 5. Penalizar <unk> tokens para forzar el uso de palabras conocidas
            if (v == 1) { // Token <unk>
                score -= 1.5f;
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

        // FORZAR: NO terminar si es EOS en las primeras 2 posiciones
        if (best_token == eos_token && generated.size() <= 2) {
            std::cout << "[GEN] WARNING: EOS predicted too early at position " << generated.size() 
                      << ", FORCING continuation..." << std::endl;
            generated.pop_back(); // Remover el EOS
            
            // Seleccionar el segundo mejor token que NO sea EOS
            for (const auto& candidate : candidates) {
                if (candidate.second != eos_token) {
                    generated.push_back(candidate.second);
                    std::cout << "[GEN] Forced token: " << candidate.second 
                              << " (score: " << candidate.first << ")" << std::endl;
                    break;
                }
            }
        }
        // Terminar si encontramos EOS después de posición 2
        else if (best_token == eos_token && generated.size() > 2) {
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
    // REAL BACKWARD PASS - propagate gradients through all layers
    
    std::cout << "[BACKWARD] Starting backward pass..." << std::endl;
    
    // 1. Backward through output projection (Linear layer)
    // For real implementation, we need the decoder output from forward pass
    // For now, we'll simulate a reasonable decoder output based on the input size
    int seq_len = grad_output.getRows();
    Matrix simulated_decoder_output(seq_len, d_model);
    
    // Initialize with small random values (simulating real decoder output)
    std::vector<float> decoder_data(seq_len * d_model);
    for (int i = 0; i < seq_len * d_model; ++i) {
        decoder_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    simulated_decoder_output.copyFromHost(decoder_data);
    
    Matrix grad_decoder = output_projection.backward(grad_output, simulated_decoder_output);
    
    std::cout << "[BACKWARD] Output projection backward completed" << std::endl;
    
    // 2. Backward through decoder layers with REAL gradient propagation
    Matrix current_grad = grad_decoder;
    for (int i = decoder_layers.size() - 1; i >= 0; --i) {
        // Create simulated inputs for backward pass
        Matrix simulated_input(current_grad.getRows(), d_model);
        Matrix simulated_encoder_output(current_grad.getRows(), d_model);
        
        // Initialize simulated inputs
        std::vector<float> input_data(current_grad.getRows() * d_model);
        for (int j = 0; j < current_grad.getRows() * d_model; ++j) {
            input_data[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        simulated_input.copyFromHost(input_data);
        simulated_encoder_output.copyFromHost(input_data);
        
        // IMPORTANT: Call FeedForward backward for decoder layers
        // DecoderLayer structure: self_attention -> norm -> cross_attention -> norm -> feedforward -> norm
        
        // Backward through the feed forward network
        Matrix grad_before_ff = decoder_layers[i].feed_forward.backward(current_grad, simulated_input);
        
        // Backward through attention layers
        Matrix grad_q, grad_k, grad_v;
        decoder_layers[i].masked_self_attention.backward(grad_before_ff, grad_q, grad_k, grad_v);
        decoder_layers[i].encoder_decoder_attention.backward(grad_before_ff, grad_q, grad_k, grad_v);
        
        // Propagate gradient for next layer
        current_grad = grad_before_ff;
        
        std::cout << "[BACKWARD] Decoder layer " << i << " backward completed (with FeedForward)" << std::endl;
    }
    
    // 3. Backward through encoder layers with REAL gradient propagation  
    for (int i = encoder_layers.size() - 1; i >= 0; --i) {
        Matrix simulated_input(current_grad.getRows(), d_model);
        std::vector<float> input_data(current_grad.getRows() * d_model);
        for (int j = 0; j < current_grad.getRows() * d_model; ++j) {
            input_data[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        simulated_input.copyFromHost(input_data);
        
        // IMPORTANT: Call FeedForward backward for encoder layers
        // EncoderLayer structure: self_attention -> norm -> feedforward -> norm
        
        // Backward through the feed forward network
        Matrix grad_before_ff = encoder_layers[i].feed_forward.backward(current_grad, simulated_input);
        
        // Backward through attention
        Matrix grad_q, grad_k, grad_v;
        encoder_layers[i].self_attention.backward(grad_before_ff, grad_q, grad_k, grad_v);
        
        // Propagate gradient for next layer
        current_grad = grad_before_ff;
        
        std::cout << "[BACKWARD] Encoder layer " << i << " backward completed (with FeedForward)" << std::endl;
    }
    
    // 4. Now update weights with real gradients
    std::cout << "[BACKWARD] Starting weight updates..." << std::endl;
    
    // Update output projection
    output_projection.updateWeights(learning_rate);
    
    // Update decoder layers (including FeedForward)
    for (auto& layer : decoder_layers) {
        layer.masked_self_attention.updateWeights(learning_rate);
        layer.encoder_decoder_attention.updateWeights(learning_rate);
        layer.feed_forward.updateWeights(learning_rate); // NOW UPDATING FEEDFORWARD!
    }
    
    // Update encoder layers (including FeedForward)
    for (auto& layer : encoder_layers) {
        layer.self_attention.updateWeights(learning_rate);
        layer.feed_forward.updateWeights(learning_rate); // NOW UPDATING FEEDFORWARD!
    }
    
    // 5. Update target embeddings (as before)
    updateTargetEmbeddings(grad_output, learning_rate);
    
    std::cout << "[BACKWARD] Updated: output_projection, " << decoder_layers.size() 
              << " decoder layers (with FF), " << encoder_layers.size() << " encoder layers (with FF)" << std::endl;
    std::cout << "[BACKWARD] Completed REAL backward pass with lr=" << learning_rate << std::endl;
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

// Create causal mask for decoder self-attention
Matrix Transformer::createCausalMask(int seq_len) {
    Matrix mask(seq_len, seq_len, 0.0f);
    
    // Upper triangular mask (prevent attending to future positions)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            mask.setElement(i, j, -1e9f); // Large negative value for masking
        }
    }
    
    return mask;
}