# FASE 1 - Paso 1: Carga del dataset CIFAR-10

from tensorflow.keras.datasets import cifar10

# Cargar los datos (imágenes y etiquetas)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Tamaño del conjunto de entrenamiento:", x_train.shape)
print("Tamaño del conjunto de prueba:", x_test.shape)
