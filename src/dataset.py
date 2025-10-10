# FASE 1 - Paso 1: Carga del dataset CIFAR-10

from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos (imágenes y etiquetas)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Tamaño del conjunto de entrenamiento:", x_train.shape)
print("Tamaño del conjunto de prueba:", x_test.shape)

# Definir las etiquetas de las 10 clases
class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Mostrar una cuadrícula con 9 imágenes de ejemplo
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis('off')
plt.tight_layout()
plt.show()