import os
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = None
history_data = None


# --- Configuración general ---
st.set_page_config(page_title="CNN CIFAR-10", page_icon="🧠", layout="wide")
st.title("🧠 Clasificación de Imágenes CIFAR-10 con CNN")
st.markdown("**Autores:** Daniel Serrano y Alexander Arrosquipa — Universidad UNIE")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_cifar10_trained.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Cargar datos ---
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# --- Función para crear el modelo ---
def create_model():
    model = Sequential(name="CNN_CIFAR10_Base")
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), name="Conv2D_1"))
    model.add(MaxPooling2D((2,2), name="MaxPool_1"))
    model.add(Conv2D(64, (3,3), activation='relu', name="Conv2D_2"))
    model.add(MaxPooling2D((2,2), name="MaxPool_2"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(64, activation='relu', name="Dense_64"))
    model.add(Dense(10, activation='softmax', name="Output"))
    return model

model = None  # Inicializamos el modelo por seguridad

# --- Cargar modelo existente o entrenar ---
if os.path.exists(MODEL_PATH):
    st.success("✅ Modelo entrenado encontrado. Cargando desde disco...")
    model = load_model(MODEL_PATH)

    # ⚙️ Asegurar compilación (necesario para evaluar o predecir)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history_data = json.load(f)
        st.info("📊 Historial de entrenamiento cargado.")
    else:
        history_data = None


# --- Botón para entrenar el modelo (solo si no existe) ---
if not os.path.exists(MODEL_PATH):
    if st.button("🚀 Entrenar modelo (8 épocas)"):
        st.info("🏋️ Entrenando modelo, por favor espera...")

        # Asegurar que el modelo existe antes de compilar
        if model is None:
            model = create_model()

        model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(
            x_train, y_train,
            epochs=8,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )

        st.success("✅ Entrenamiento completado. Guardando modelo...")
        model.save(MODEL_PATH)

        with open(HISTORY_PATH, "w") as f:
            json.dump(history.history, f)

        history_data = history.history
        st.rerun()  # 🔁 recarga la app automáticamente tras entrenar
else:
    st.write("📂 Modelo ya entrenado y cargado correctamente.")

# --- Evaluación ---
if model is not None:
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.metric(label="📈 Precisión final en test", value=f"{test_acc:.2%}")
    st.metric(label="📉 Pérdida en test", value=f"{test_loss:.4f}")


# --- Gráficas ---
if history_data is not None:
    acc = history_data.get('accuracy', [])
    val_acc = history_data.get('val_accuracy', [])
    loss = history_data.get('loss', [])
    val_loss = history_data.get('val_loss', [])

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    axes[0].plot(acc, label='Entrenamiento')
    axes[0].plot(val_acc, label='Validación')
    axes[0].set_title('Precisión')
    axes[0].legend()

    axes[1].plot(loss, label='Entrenamiento')
    axes[1].plot(val_loss, label='Validación')
    axes[1].set_title('Pérdida')
    axes[1].legend()

    st.pyplot(fig)

# --- Predicción de ejemplo (solo si hay modelo cargado) ---
if model is not None:
    st.subheader("🔍 Prueba una imagen aleatoria del conjunto de test")
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    pred = model.predict(img[np.newaxis, ...])
    predicted_class = class_names[np.argmax(pred)]
    true_class = class_names[np.argmax(y_test[idx])]

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=f"Verdadero: {true_class}", use_container_width=True)
    with col2:
        st.write(f"**Predicción del modelo:** {predicted_class}")
else:
    st.info("⚠️ El modelo aún no ha sido cargado ni entrenado. Entrénalo para realizar predicciones.")
