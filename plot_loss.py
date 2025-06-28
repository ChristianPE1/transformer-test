#!/usr/bin/env python3
"""
Script para visualizar la pérdida del entrenamiento del transformer.
Uso: python plot_loss.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_loss():
    """Lee el archivo training_loss.txt y genera gráficos de la pérdida."""
    
    # Verificar si el archivo existe
    if not os.path.exists('training_loss.txt'):
        print("Error: No se encontró el archivo training_loss.txt")
        print("Asegúrate de ejecutar el entrenamiento primero.")
        return
    
    try:
        # Leer los datos
        df = pd.read_csv('training_loss.txt')
        print(f"Datos cargados: {len(df)} épocas")
        
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis de Entrenamiento del Transformer', fontsize=16)
        
        # 1. Pérdida a lo largo del tiempo
        ax1.plot(df['Epoch'], df['Loss'], 'b-', linewidth=2, label='Pérdida')
        ax1.plot(df['Epoch'], df['BestLoss'], 'r--', linewidth=2, label='Mejor pérdida')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.set_title('Evolución de la Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pérdida en escala logarítmica
        ax2.semilogy(df['Epoch'], df['Loss'], 'g-', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida (escala log)')
        ax2.set_title('Pérdida en Escala Logarítmica')
        ax2.grid(True, alpha=0.3)
        
        # 3. Progreso de mejora (ventana móvil)
        window_size = min(10, len(df) // 4)
        if window_size > 1:
            rolling_mean = df['Loss'].rolling(window=window_size).mean()
            ax3.plot(df['Epoch'], df['Loss'], 'lightblue', alpha=0.5, label='Pérdida original')
            ax3.plot(df['Epoch'], rolling_mean, 'darkblue', linewidth=2, label=f'Media móvil ({window_size} épocas)')
            ax3.set_xlabel('Época')
            ax3.set_ylabel('Pérdida')
            ax3.set_title('Tendencia Suavizada')
            ax3.legend()
        else:
            ax3.plot(df['Epoch'], df['Loss'], 'b-', linewidth=2)
            ax3.set_title('Pérdida (sin suavizado)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Épocas estancadas
        ax4.plot(df['Epoch'], df['StagnantEpochs'], 'orange', linewidth=2)
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Épocas sin mejora')
        ax4.set_title('Épocas Estancadas')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar la figura
        plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
        print("Gráfico guardado como 'training_loss_plot.png'")
        
        # Mostrar estadísticas
        print("\n=== ESTADÍSTICAS DE ENTRENAMIENTO ===")
        print(f"Pérdida inicial: {df['Loss'].iloc[0]:.6f}")
        print(f"Pérdida final: {df['Loss'].iloc[-1]:.6f}")
        print(f"Mejor pérdida: {df['BestLoss'].min():.6f}")
        print(f"Mejora total: {df['Loss'].iloc[0] - df['Loss'].iloc[-1]:.6f}")
        print(f"Mejora porcentual: {((df['Loss'].iloc[0] - df['Loss'].iloc[-1]) / df['Loss'].iloc[0]) * 100:.2f}%")
        print(f"Máximas épocas estancadas: {df['StagnantEpochs'].max()}")
        
        # Detectar si hay problemas
        if df['Loss'].iloc[-1] > df['Loss'].iloc[0] * 0.95:
            print("\n⚠️  ADVERTENCIA: La pérdida no ha mejorado significativamente")
            print("   Considera ajustar el learning rate, aumentar épocas, o revisar la implementación")
        
        if df['StagnantEpochs'].max() > len(df) * 0.3:
            print("\n⚠️  ADVERTENCIA: Muchas épocas sin mejora detectadas")
            print("   El modelo podría estar convergiendo prematuramente o tener problemas de optimización")
        
        # Mostrar el gráfico
        plt.show()
        
    except Exception as e:
        print(f"Error al procesar los datos: {e}")

if __name__ == "__main__":
    plot_training_loss()
