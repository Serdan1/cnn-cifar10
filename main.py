import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from src.dataset import x_train, y_train, x_test, y_test
from src.cnn_model import model

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_cifar10_trained.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

# Crear carpeta de modelos si no existe
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔍 Verificar si el modelo ya está entrenado
if os.path.exists(MODEL_PATH):
    print("✅ Modelo entrenado encontrado. Cargando desde disco...")
    model = load_model(MODEL_PATH)

    # Cargar historial si existe
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history_data = json.load(f)
        print("📊 Historial de entrenamiento cargado.")
    else:
        history_data = None

else:
    print("🏋️ Entrenando modelo (primer uso)...")

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        epochs=8,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    print("💾 Guardando modelo y resultados...")
    model.save(MODEL_PATH)

    # Guardar historial de entrenamiento
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f)
    history_data = history.history

# 📈 Evaluación del modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"📊 Precisión final en test: {test_acc:.2f}")

# (Opcional) Mostrar resumen visual básico si hay historial
if history_data:
    import matplotlib.pyplot as plt

    acc = history_data.get('accuracy', [])
    val_acc = history_data.get('val_accuracy', [])
    loss = history_data.get('loss', [])
    val_loss = history_data.get('val_loss', [])

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label='Entrenamiento')
    plt.plot(val_acc, label='Validación')
    plt.title('Precisión'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loss, label='Entrenamiento')
    plt.plot(val_loss, label='Validación')
    plt.title('Pérdida'); plt.legend()
    plt.tight_layout()
    plt.show()
