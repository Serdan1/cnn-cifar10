# src/train_and_evaluate.py

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from cnn_model import model

# 🧠 Cargar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 🔄 Normalizar los valores de píxeles (0-255 → 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 🏷️ Convertir etiquetas a formato one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ⚙️ COMPILACIÓN DEL MODELO
# 'adam': optimizador eficiente y adaptativo
# 'categorical_crossentropy': pérdida para clasificación multiclase
# 'accuracy': métrica principal de rendimiento
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo compilado correctamente.")
model.summary()

# 🧩 ENTRENAMIENTO DEL MODELO

# Entrenar el modelo durante 10 épocas, usando 10% de los datos para validación
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Guardar el historial para análisis posterior
print("✅ Entrenamiento completado.")

# 🧮 EVALUACIÓN DEL MODELO

# Evaluar el modelo sobre el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f"\n📊 Resultados en el conjunto de prueba:")
print(f"   Pérdida (loss): {test_loss:.4f}")
print(f"   Precisión (accuracy): {test_accuracy:.4f}")


