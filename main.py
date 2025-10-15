"""
📦 Proyecto: Clasificación de Imágenes con CNN (CIFAR-10)
Autor: Daniel serrano y Alexander Arrosquipa
Universidad: UNIE
Descripción: Script principal que ejecuta todo el flujo del sistema:
             carga de datos, creación del modelo, entrenamiento y evaluación.
"""

from src.dataset import x_train, y_train, x_test, y_test
from src.cnn_model import model

# 🔧 Compilar el modelo
print("\n🔧 Compilando el modelo...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("✅ Modelo compilado correctamente.")

# 🚀 Entrenar el modelo
print("\n🚀 Iniciando entrenamiento...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 🧪 Evaluar el modelo
print("\n🧪 Evaluando el modelo en el conjunto de prueba...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"\n📊 Resultados finales:")
print(f"   Pérdida (loss): {test_loss:.4f}")
print(f"   Precisión (accuracy): {test_accuracy:.4f}")

# 📈 Mostrar las curvas de entrenamiento
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Entrenamiento')
plt.plot(epochs, val_acc, 'ro-', label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Entrenamiento')
plt.plot(epochs, val_loss, 'ro-', label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
