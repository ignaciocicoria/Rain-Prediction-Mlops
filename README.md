#  Rain Prediction Model – Australia

Proyecto de *Machine Learning* enfocado en la predicción de lluvia en distintas regiones de Australia, a partir de datos meteorológicos históricos.  
Incluye análisis exploratorio, preprocesamiento avanzado, modelado con técnicas de clasificación y redes neuronales, optimización de hiperparámetros, interpretabilidad con SHAP y despliegue del modelo mediante Docker.

---

##  Objetivo

Desarrollar un modelo predictivo capaz de anticipar si lloverá al día siguiente (`RainTomorrow`) utilizando el dataset `weatherAUS.csv`.  
El flujo completo abarca desde el análisis y preparación de datos hasta la puesta en producción del modelo, aplicando prácticas de **MLOps**.

---

##  Estructura del proyecto

```
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
```
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

## 📈 Resultados y métricas del modelo final

 
El modelo final —que combina **regularización**, **ajuste de umbral** y **optimización de hiperparámetros**— logró el mejor equilibrio entre recall y precisión, alineándose con el objetivo del problema: **maximizar la detección de lluvias**.

Este enfoque prioriza el **recall de la clase minoritaria (lluvia)**, aun sacrificando cierta precisión.  
En términos prácticos, el modelo detecta correctamente el 92 % de los días en los que efectivamente llueve.

| Clase | Precision | Recall | F1-score | Support |
|:------|-----------:|--------:|----------:|--------:|
| 0 (No lluvia) | 0.98 | 0.34 | 0.50 | 22 064 |
| 1 (Lluvia) | 0.30 | 0.98 | 0.46 | 6 375 |
| **Accuracy global** |   |   | **0.48** | 28 439 |
| **Macro promedio** | 0.64 | 0.66 | 0.48 | 28 439 |
| **Weighted promedio** | 0.83 | 0.48 | 0.49 | 28 439 |

Además, se definió una **métrica custom (custom = 0.75)** para ponderar el recall de la clase 1, que representa los casos en los que realmente llueve.  
Los modelos de AutoML y redes neuronales obtuvieron valores similares (`custom = 0.71`), aunque el modelo optimizado sigue siendo el más robusto para este objetivo.

**Conclusión:**  
El ajuste del umbral de decisión y la regularización permiten construir un modelo más sensible a los casos de lluvia, priorizando la detección (recall) sobre la precisión.  
Según el caso de uso, esta preferencia puede ajustarse para lograr un balance diferente entre ambos indicadores.

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
