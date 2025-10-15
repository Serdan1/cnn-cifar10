import streamlit as st
from src.dataset import x_train, y_train, x_test, y_test
from src.cnn_model import model
import matplotlib.pyplot as plt

st.title("🧠 Clasificación de Imágenes con CNN — CIFAR-10")

if st.button("🚀 Ejecutar entrenamiento y evaluación del modelo"):
    st.write("🔧 Compilando modelo...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    st.write("🏋️ Entrenando modelo...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    
    st.write("🧪 Evaluando modelo...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.success(f"✅ Precisión final: {test_acc:.2f}")
    
    # Gráficas
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Entrenamiento')
    ax[0].plot(history.history['val_accuracy'], label='Validación')
    ax[0].set_title('Precisión')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Entrenamiento')
    ax[1].plot(history.history['val_loss'], label='Validación')
    ax[1].set_title('Pérdida')
    ax[1].legend()
    
    st.pyplot(fig)
