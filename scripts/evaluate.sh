#!/bin/bash
# filepath: scripts/evaluate.sh

echo "=== Evaluando Transformer CUDA ==="

# Cambiar al directorio del proyecto
cd "$(dirname "$0")/.."

# Verificar que el executable existe
if [ ! -f "bin/cuda_transformer" ]; then
    echo "Error: No se encontró bin/cuda_transformer"
    echo "Ejecuta 'make' para compilar el proyecto primero"
    exit 1
fi

echo "Ejecutando evaluación..."
./bin/cuda_transformer

echo "¡Evaluación completada!"