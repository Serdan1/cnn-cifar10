# cnn-cifar10
Implementación de una Red Neuronal Convolucional para clasificación de imágenes CIFAR-100 con TensorFlow/Keras
# 🧠 Clasificación de Imágenes con CNN — CIFAR-10

Proyecto académico desarrollado como parte de la asignatura **Lenguaje Natural y Compiladores** (Universidad UNIE).  
El objetivo es implementar y entrenar una **Red Neuronal Convolucional (CNN)** capaz de reconocer objetos en imágenes del conjunto de datos **CIFAR-10**, utilizando **TensorFlow y Keras**.

---

## 🚀 1. Descripción del Proyecto

Este proyecto busca construir un **“córtex visual” artificial**, inspirado en el cerebro humano, que pueda identificar patrones visuales como bordes, texturas y formas dentro de imágenes a color.

El modelo se entrena con **CIFAR-10**, un dataset que contiene 60.000 imágenes (32x32 píxeles) distribuidas en 10 categorías:
> Avión ✈️, Automóvil 🚗, Pájaro 🐦, Gato 🐱, Ciervo 🦌, Perro 🐶, Rana 🐸, Caballo 🐴, Barco ⛵ y Camión 🚚.

---

## 🧩 2. Arquitectura del Sistema

El proyecto está estructurado modularmente en 3 componentes principales:

cnn-cifar10/
│
├── src/
│   ├── dataset.py             # Carga y preprocesamiento del dataset CIFAR-10
│   ├── cnn_model.py           # Definición de la arquitectura CNN
│   └── train_and_evaluate.py  # Entrenamiento, evaluación y visualización
│
├── main.py                    # Punto de entrada principal del sistema
├── requirements.txt           # Dependencias del proyecto
└── docs/
    └── Informe_Tecnico_CNN.pdf  # Informe final del proyecto

---

## 🧱 3. Arquitectura del Modelo CNN

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

## 🧠 4. Instalación y Requisitos

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


## 🚀 5. Ejecución del Sistema
Para ejecutar todo el flujo del proyecto (carga, entrenamiento, evaluación y gráficas):
python main.py

Esto mostrará:

El entrenamiento de la CNN (10 épocas)

La precisión final en el conjunto de prueba (~70%)

Las gráficas de pérdida y precisión del modelo

## 📊 6. Resultados Obtenidos
Métrica	Valor
Precisión de entrenamiento	0.76
Precisión de validación	0.71
Precisión en test	0.70
Pérdida en test	0.85

📈 Las curvas de entrenamiento muestran una convergencia estable sin sobreajuste.


## 🧭 7. Conclusiones

El modelo logra una precisión del 70% en el conjunto de prueba,
demostrando la capacidad de las CNN para extraer automáticamente características visuales
y superar los métodos tradicionales en tareas de visión artificial.

Mejoras futuras:

Añadir Dropout o Data Augmentation

Aumentar las épocas de entrenamiento

Usar Batch Normalization
