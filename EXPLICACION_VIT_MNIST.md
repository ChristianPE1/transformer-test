# Explicación paso a paso del Vision Transformer (ViT) para MNIST

Este documento describe detalladamente el flujo de datos, las clases y funciones principales, y el propósito de cada variable relevante en tu implementación de Vision Transformer para clasificación de dígitos MNIST.

---

## 1. Flujo General del Programa

### 1.1. Inicialización
- El programa comienza en `main.cu`.
- Se verifica la disponibilidad de CUDA.
- Se carga el dataset MNIST usando la clase `MNISTLoader`.
- Se instancia el modelo `ViTMNIST` con los parámetros deseados.

### 1.2. Entrenamiento
- Para cada época y cada imagen:
  - Se convierte la imagen a un objeto `Matrix`.
  - Se realiza el forward pass por el ViT.
  - Se calcula la pérdida (cross-entropy) y sus gradientes.
  - Se realiza el backward pass para propagar los gradientes.
  - Se actualizan los pesos del modelo.

---

## 2. Clases y Funciones Principales

### 2.1. `MNISTLoader`
- **Propósito:** Cargar imágenes y etiquetas del dataset MNIST.
- **Métodos:**
  - `load`: Lee archivos binarios y devuelve un struct con imágenes (vector de float) y etiquetas (vector de int).

### 2.2. `ViTMNIST`
- **Propósito:** Implementa el Vision Transformer completo para clasificación.
- **Variables:**
  - `patch_embed`: Instancia de `PatchEmbedding` para dividir la imagen en patches y proyectarlos.
  - `blocks`: Vector de `ViTBlock`, cada uno es un bloque transformer encoder.
  - `norm`: Capa de normalización final.
  - `classifier`: Capa lineal para clasificación (128 → 10).
  - `pos_embedding`: Embeddings posicionales para los patches.
  - `num_patches`, `embed_dim`, `num_classes`: Hiperparámetros del modelo.
  - `last_pooled`, `last_normalized`: Guardan los resultados intermedios del forward para el backward.
- **Métodos:**
  - `forward`: Ejecuta el paso hacia adelante por todo el modelo.
  - `backward`: Propaga los gradientes hacia atrás por todos los componentes.
  - `update_weights`: Actualiza los pesos de todos los componentes.

### 2.3. `PatchEmbedding`
- **Propósito:** Divide la imagen en patches y los proyecta a un espacio de embedding.
- **Variables:**
  - `patch_size`: Tamaño de cada patch (ej. 4).
  - `embed_dim`: Dimensión del embedding.
  - `projection`: Matriz de pesos para proyectar cada patch.
- **Métodos:**
  - `forward`: Convierte la imagen 28x28 en una matriz de patches y la proyecta a `embed_dim`.

### 2.4. `ViTBlock`
- **Propósito:** Un bloque transformer encoder.
- **Variables:**
  - `attention`: Instancia de `MultiHeadAttention`.
  - `mlp`: FeedForward (MLP) de dos capas.
  - `norm1`, `norm2`: Capas de normalización.
- **Métodos:**
  - `forward`: Aplica atención, normalización, MLP y residuals.
  - `backward`: Propaga gradientes hacia atrás por el bloque.
  - `updateWeights`: Actualiza los pesos del bloque.

### 2.5. `MultiHeadAttention`
- **Propósito:** Permite que cada patch "vea" a todos los demás.
- **Variables:**
  - Pesos para queries, keys, values, y proyección final.
- **Métodos:**
  - `forward`: Calcula la atención multi-cabeza.
  - `backward`: Calcula gradientes para los pesos de atención.
  - `updateWeights`: Actualiza los pesos.

### 2.6. `FeedForward`
- **Propósito:** MLP de dos capas para cada patch.
- **Variables:**
  - `W1`, `W2`: Matrices de pesos.
  - `b1`, `b2`: Bias.
- **Métodos:**
  - `forward`: Aplica dos capas lineales y ReLU.
  - `backward`: Calcula gradientes para los pesos.
  - `updateWeights`: Actualiza los pesos.

### 2.7. `LayerNorm`
- **Propósito:** Normaliza las activaciones por patch.
- **Variables:**
  - Parámetros de escala y desplazamiento.
- **Métodos:**
  - `forward`: Normaliza la entrada.
  - `backward`: Calcula gradientes para los parámetros.
  - `updateWeights`: Actualiza los parámetros.

### 2.8. `Linear`
- **Propósito:** Capa lineal final para clasificación.
- **Variables:**
  - `weights`, `bias`: Pesos y bias.
- **Métodos:**
  - `forward`: Multiplicación matriz-vector.
  - `backward`: Gradientes para pesos y bias.
  - `updateWeights`: Actualiza los pesos.

### 2.9. `CrossEntropyLoss`
- **Propósito:** Calcula la pérdida y gradientes para clasificación.
- **Métodos:**
  - `compute_loss`: Calcula la pérdida de entropía cruzada.
  - `compute_gradients`: Calcula gradientes para el backward.

---

## 3. Proceso Detallado de Forward y Backward

### 3.1. Forward Pass
1. **Imagen 28x28 → patches 4x4 → 49x16**
2. **Proyección lineal → 49x128**
3. **Suma de embeddings posicionales → 49x128**
4. **6 bloques ViTBlock (cada uno: atención, norm, MLP, norm)**
5. **LayerNorm final → 49x128**
6. **Global average pooling → 1x128**
7. **Clasificador lineal → 1x10**

### 3.2. Backward Pass
1. **Gradiente de la pérdida respecto a la salida del clasificador**
2. **Retropropagación por el clasificador**
3. **Expandir gradiente para el pooling**
4. **Retropropagación por LayerNorm y los bloques ViTBlock (en orden inverso)**
5. **Retropropagación por PatchEmbedding (opcional, si se implementa)**

### 3.3. Actualización de Pesos
- Cada componente actualiza sus pesos usando los gradientes calculados y el learning rate.

---

