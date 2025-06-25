#!/bin/bash
# filepath: scripts/train.sh

echo "=== Entrenando Transformer CUDA ==="

# Cambiar al directorio del proyecto
cd "$(dirname "$0")/.."

# Verificar que el executable existe
if [ ! -f "bin/cuda_transformer" ]; then
    echo "Error: No se encontró bin/cuda_transformer"
    echo "Ejecuta 'make' para compilar el proyecto primero"
    exit 1
fi

# Verificar que el archivo TSV existe
if [ ! -f "db_translate.tsv" ]; then
    echo "Error: No se encontró db_translate.tsv"
    exit 1
fi

echo "Ejecutando entrenamiento..."
./bin/cuda_transformer

echo "¡Entrenamiento completado!"