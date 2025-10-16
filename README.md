# 🧠 Clasificación de Imágenes con CNN — CIFAR-10

Proyecto académico desarrollado como parte de la asignatura **Lenguaje Natural y Compiladores** (Universidad UNIE).  
El objetivo es implementar y entrenar una **Red Neuronal Convolucional (CNN)** capaz de reconocer objetos en imágenes del conjunto de datos **CIFAR-10**, utilizando **TensorFlow y Keras**.

### Ejecuta el sistema:

#### 💻 Interfaz con Streamlit

Este proyecto incluye una interfaz interactiva desarrollada con Streamlit que permite visualizar el modelo CNN entrenado, las métricas de rendimiento y realizar predicciones en tiempo real sobre imágenes del dataset CIFAR-10.

[streamlit run app.py](https://cnn-cifar10-gksz6dmdjwxwrydjlsx2rp.streamlit.app/)

### 📓 Notebook
- [Abrir en Colab](https://colab.research.google.com/github/Serdan1/cnn-cifar10/blob/main/notebooks/cnn_cifar10.ipynb)
- O descarga el `.ipynb` desde `notebooks/` del repositorio.

---

## 🚀 1. Descripción del Proyecto

Este proyecto busca construir un **“córtex visual” artificial**, inspirado en el cerebro humano, que pueda identificar patrones visuales como bordes, texturas y formas dentro de imágenes a color.

El modelo se entrena con **CIFAR-10**, un dataset que contiene 60.000 imágenes (32x32 píxeles) distribuidas en 10 categorías:
> Avión ✈️, Automóvil 🚗, Pájaro 🐦, Gato 🐱, Ciervo 🦌, Perro 🐶, Rana 🐸, Caballo 🐴, Barco ⛵ y Camión 🚚.

---

## 🧱 2. Arquitectura del Modelo CNN

**Modelo implementado en `src/cnn_model.py`:**

| Tipo de capa | Parámetros | Descripción |
|---------------|-------------|--------------|
| Conv2D (32 filtros, 3x3) | activación ReLU | Detección de bordes y texturas simples |
| MaxPooling2D (2x2) | — | Reducción del tamaño espacial |
| Conv2D (64 filtros, 3x3) | activación ReLU | Combinación de patrones más complejos |
| MaxPooling2D (2x2) | — | Resumen de características |
| Flatten | — | Conversión a vector 1D |
| Dense (64) | activación ReLU | Integración de características extraídas |
| Dense (10) | activación Softmax | Clasificación en 10 clases |

**Total de parámetros entrenables:** `167,562`

---

## 🧠 3. Instalación y Requisitos

### 🔧 Requisitos previos
- Python 3.10 o superior  
- pip (gestor de paquetes)

### ⚙️ Instalación
# Clonar el repositorio
git clone https://github.com/Serdan1/cnn-cifar10.git
cd cnn-cifar10

# Crear entorno virtual
python -m venv venv

.\venv\Scripts\activate  # En Windows

venv/bin/activate # En Linux/Mac

# Instalar dependencias
pip install -r requirements.txt


## 🚀 4. Ejecución del Sistema
Para ejecutar todo el flujo del proyecto (carga, entrenamiento, evaluación y gráficas):
python main.py

Esto mostrará:

El entrenamiento de la CNN (10 épocas)

La precisión final en el conjunto de prueba (~70%)

Las gráficas de pérdida y precisión del modelo

## 📊 5. Resultados Obtenidos
Métrica	Valor
Precisión de entrenamiento	0.76
Precisión de validación	0.71
Precisión en test	0.70
Pérdida en test	0.85

📈 Las curvas de entrenamiento muestran una convergencia estable sin sobreajuste.


## 🧭 6. Conclusiones

El modelo logra una precisión del 70% en el conjunto de prueba,
demostrando la capacidad de las CNN para extraer automáticamente características visuales
y superar los métodos tradicionales en tareas de visión artificial.


Este proyecto ha permitido entender en profundidad:

El proceso de aprendizaje jerárquico de las CNNs.

La importancia del gradiente y la retropropagación en la optimización.

Cómo la reducción de dimensionalidad mediante pooling mejora la generalización.

El modelo alcanza una precisión del 70 % en test, lo que confirma la validez del diseño y entrenamiento implementado.


### 🧩 Arquitectura del Sistema (Mermaid)

```mermaid
graph TD
    %% --- Dataset ---
    A[Dataset CIFAR-10] -->|Carga y preprocesamiento| B[dataset.py]
    B -->|Normalizacion y One Hot Encoding| C[cnn_model.py]
    
    %% --- Modelo CNN ---
    subgraph Modelo_CNN [Arquitectura de la CNN]
        C1[Conv2D 32 filtros]
        C2[MaxPooling2D]
        C3[Conv2D 64 filtros]
        C4[MaxPooling2D]
        C5[Flatten]
        C6[Dense 64]
        C7[Dense 10 Softmax]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    end
    
    C -->|Definicion del modelo| Modelo_CNN --> D[train_and_evaluate.py]
    
    %% --- Entrenamiento y Evaluacion ---
    D -->|Entrenamiento 8 epocas| E[cnn_cifar10_trained.h5]
    D -->|Historial de entrenamiento| F[training_history.json]
    
    %% --- Evaluacion ---
    E -->|Evaluacion con test set| G[Precision y Perdida]
    
    %% --- Interfaz ---
    G -->|Visualizacion y predicciones| H[Streamlit app.py]
    F --> H
    E --> H
    
    %% --- Resultados finales ---
    H -->|Resultados visuales y predicciones| I[Usuario]
