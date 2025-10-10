# FASE 1

import ssl
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

# 🔧 Solución SSL para evitar errores de certificado (Python 3.13 en Windows)
ssl._create_default_https_context = ssl._create_unverified_context

# Cargar el dataset CIFAR-10
print("📦 Cargando el dataset CIFAR-10 (automáticamente desde Keras)...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("✅ Dataset CIFAR-10 cargado correctamente.")

# Mostrar tamaños
print("Tamaño del conjunto de entrenamiento:", x_train.shape)
print("Tamaño del conjunto de prueba:", x_test.shape)

# Definir las etiquetas de las 10 clases
class_names = [
    'avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
    'perro', 'rana', 'caballo', 'barco', 'camión'
]

# Mostrar una cuadrícula con 9 imágenes de ejemplo
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Normalizar los píxeles al rango [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertir las etiquetas a one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("\nDespués de normalizar y codificar:")
print("x_train shape:", x_train.shape, flush=True)
print("y_train shape:", y_train.shape, flush=True)
print("Ejemplo de etiqueta codificada:", y_train[0], flush=True)
