"""
FASE 2: Construcción de la arquitectura base de la CNN para CIFAR-10.
Autor: Alexa
Descripción:
    Este script define la arquitectura base de una red neuronal convolucional (CNN)
    utilizando TensorFlow/Keras. El modelo se compone de dos bloques convolucionales
    para extracción de características y un clasificador totalmente conectado.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 🧩 Inicializar el modelo secuencial
model = Sequential(name="CNN_CIFAR10_Base")

# 🧠 BLOQUE CONVOLUCIONAL 1
# Conv2D: Aprende 32 filtros de 3x3 que extraen características locales (bordes, colores, texturas)
# input_shape: tamaño de las imágenes (32x32 píxeles con 3 canales RGB)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name="Conv2D_1"))
# MaxPooling2D: reduce la dimensión espacial, manteniendo las características más importantes
model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_1"))

# 🔎 BLOQUE CONVOLUCIONAL 2
# Conv2D: Aprende 64 filtros de 3x3 más complejos (combinaciones de patrones detectados por el bloque anterior)
model.add(Conv2D(64, (3, 3), activation='relu', name="Conv2D_2"))
# MaxPooling2D: vuelve a reducir la dimensión espacial
model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_2"))

# 🧮 CLASIFICADOR (Fully Connected)
# Flatten: convierte los mapas de características 2D en un vector 1D
model.add(Flatten(name="Flatten"))
# Dense(64): capa densa intermedia que combina las características extraídas
model.add(Dense(64, activation='relu', name="Dense_64"))
# Dense(10): capa de salida con 10 neuronas (una por clase), usando softmax
model.add(Dense(10, activation='softmax', name="Output"))

# 📋 Mostrar resumen del modelo
model.summary()
