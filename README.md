#  Rain Prediction Model – Australia

Proyecto de *Machine Learning* enfocado en la predicción de lluvia en distintas regiones de Australia, a partir de datos meteorológicos históricos.  
Incluye análisis exploratorio, preprocesamiento avanzado, modelado con técnicas de clasificación y redes neuronales, optimización de hiperparámetros, interpretabilidad con SHAP y despliegue del modelo mediante Docker.

---

##  Objetivo

Desarrollar un modelo predictivo capaz de anticipar si lloverá al día siguiente (`RainTomorrow`) utilizando el dataset `weatherAUS.csv`.  
El flujo completo abarca desde el análisis y preparación de datos hasta la puesta en producción del modelo, aplicando prácticas de **MLOps**.

---

##  Estructura del proyecto

'''
rain-prediction-mlops/
│
├── docker/ # Archivos para deployment con Docker
│ ├── Dockerfile # Instrucciones de build & run
│ ├── README.md # Documentación Docker
│ ├── inferencia.py # Script de inferencia
│ ├── requirements.txt # Librerías mínimas para inferencia
│ ├── pipeline.pkl # Pipeline serializado
│ └── transformadores/ # Módulos custom de preprocesado
│
├── TP-clasificacion-AA1.ipynb # Notebook principal de análisis y modelado
├── MLOps.ipynb # Notebook de deployment e integración con Docker
├── weatherAUS.csv # Dataset meteorológico original
├── coordenadas_aus.csv # Coordenadas geográficas para clustering
└── README.md # Documentación general del proyecto
'''
---

##  Flujo del proyecto

### 1. Exploración y análisis de datos
- Análisis descriptivo y exploratorio (EDA).  
- Visualizaciones interactivas (histogramas, scatterplots, boxplots).  
- Segmentación de ciudades en **regiones** mediante clustering.  
- Identificación y tratamiento de valores faltantes y outliers.  
- Codificación y escalado de variables numéricas y categóricas.

### 2. Modelado
- Entrenamiento con **Regresión Logística** como modelo base.  
- Evaluación con métricas: *Recall*, *Precision*, *F1 Score*, *ROC-AUC*.  
- Análisis de **falsos positivos/negativos** mediante matriz de confusión.  
- Tratamiento de desbalance de clases.

### 3. Optimización y validación
- **Grid Search** y **Optuna** para optimización de hiperparámetros.  
- Validación cruzada (K-Folds).  
- Comparación de desempeño entre modelos.

### 4. Explicabilidad
- Interpretabilidad del modelo mediante **SHAP**.  
- Visualización de impacto global y local de las variables.  
- Identificación de variables con mayor influencia en la predicción.

### 5. AutoML
- Implementación de **PyCaret** para automatizar el flujo de modelado y comparar resultados.  
- Evaluación automática de múltiples algoritmos de clasificación.

### 6. Redes Neuronales
- Implementación de un modelo **denso con TensorFlow/Keras**.  
- Ajuste de arquitectura y optimización de hiperparámetros.  
- Comparación con modelos tradicionales.  
- Evaluación de overfitting/underfitting mediante curvas de entrenamiento.

### 7. MLOps y despliegue
- Serialización del pipeline con `joblib`.  
- Construcción y ejecución de contenedor **Docker**.  
- Pruebas de inferencia en batch y sobre instancias individuales.

---

##  Tecnologías utilizadas

| Categoría | Herramientas |
|------------|---------------|
| **Lenguaje principal** | Python 3.11 |
| **Análisis y EDA** | pandas, numpy, matplotlib, seaborn, plotly |
| **Modelado ML** | scikit-learn, imbalanced-learn |
| **Optimización** | optuna |
| **Interpretabilidad** | shap |
| **AutoML** | pycaret |
| **Deep Learning** | TensorFlow / Keras |
| **Despliegue (MLOps)** | Docker |

---

## Resultados principales

- Modelo final: **Regresión Logística optimizada**  
- Métricas obtenidas:
  - **Recall:** alto desempeño en identificación de días lluviosos  
  - **Precision y F1-Score:** balanceadas según el umbral óptimo determinado  
  - **ROC-AUC:** validación del poder discriminativo del modelo  

La interpretación con SHAP permitió identificar las variables más influyentes:  
**Humidity3pm**, **RainToday**, **Pressure9am**, y **Cloud3pm**.

---

##  Deployment con Docker

El modelo final fue empacado en un contenedor Docker listo para inferencia.  
Dentro de `docker/` se incluyen:

- `Dockerfile`  
- `inferencia.py` (script predictivo)  
- `requirements.txt` (dependencias mínimas)

### Comandos principales

```bash
# Construir la imagen
docker build -t rain-predictor .

# Ejecutar el contenedor
docker run --rm rain-predictor
