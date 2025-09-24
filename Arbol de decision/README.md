# Clasificación de Spam/Ham usando Árboles de Decisión

Este proyecto implementa un clasificador de correos electrónicos usando árboles de decisión para distinguir entre mensajes SPAM y HAM (correos legítimos). El análisis incluye evaluación de consistencia del modelo mediante múltiples ejecuciones y visualización de métricas de rendimiento.

## Autores
- Maira Alejandra Chila
- Nicolás Posada García

## 📋 Contenido del Proyecto

- **arbolDecision.py**: Script principal con el modelo de clasificación
- **dataset_sintetico_spam_ham.csv**: Dataset sintético con características de correos electrónicos
- **README.md**: Documentación del proyecto

## Descripción del Código

El archivo `arbolDecision.py` realiza las siguientes tareas:

1. **Carga del dataset**: Lee el archivo CSV con características de correos electrónicos
2. **Generación de etiquetas**: Crea automáticamente las etiquetas SPAM/HAM basadas en criterios específicos
3. **Entrenamiento del modelo**: Implementa un árbol de decisión con 50 ejecuciones independientes
4. **Evaluación de métricas**: Calcula accuracy, F1-score y análisis estadísticos
5. **Análisis de consistencia**: Evalúa la estabilidad del modelo a través de múltiples ejecuciones
6. **Visualización**: Genera gráficos completos del comportamiento del modelo

## Ejecución

Ejecuta el script desde la terminal:

```bash
python arbolDecision.py
```

El programa ejecutará automáticamente 50 iteraciones del modelo y mostrará los resultados estadísticos junto con visualizaciones.

## 📊 Dataset

El dataset contiene **500 registros** con las siguientes características:

### Características de entrada:
- **longitud_cuerpo**: Longitud del contenido del correo
- **num_adjuntos**: Número de archivos adjuntos
- **num_links**: Número de enlaces en el correo
- **remitente_sospechoso**: Indicador binario (0/1) de remitente sospechoso
- **contiene_palabras_dinero**: Indicador binario de palabras relacionadas con dinero
- **urgencia**: Indicador binario de urgencia en el mensaje

### Metadatos (no usados en clasificación):
- **nombre**: Nombre del remitente
- **email**: Dirección de correo del remitente
- **fecha**: Fecha del mensaje
- **asunto**: Asunto del correo

### Variable objetivo:
- **label**: Etiqueta generada automáticamente
  - **1 (SPAM)**: Si cumple alguna condición:
    - remitente_sospechoso = 1
    - contiene_palabras_dinero = 1
    - urgencia = 1
    - num_links > 2
  - **0 (HAM)**: En caso contrario

## 🛠️ Tecnologías y Librerías

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
```

## 🔧 Funcionalidades

### 1. Preprocesamiento de Datos
- Carga del dataset desde archivo CSV
- Generación automática de etiquetas SPAM/HAM
- Selección de características relevantes para el modelo

### 2. Modelo de Machine Learning
- **Algoritmo**: Árbol de Decisión (DecisionTreeClassifier)
- **Criterio**: Gini
- **División**: 70% entrenamiento, 30% prueba
- **Validación**: 50 ejecuciones independientes

### 3. Métricas de Evaluación
- **Accuracy**: Precisión general del modelo
- **F1-Score**: Balance entre precisión y recall
- **Z-Score**: Normalización estadística de las métricas

### 4. Análisis de Estabilidad
- Medias acumuladas para evaluar convergencia
- Análisis de diferencias entre primeras y últimas ejecuciones
- Estadísticas descriptivas (media, desviación estándar, rango)

## 📈 Visualizaciones

El script genera un dashboard con 4 gráficos:

1. **Evolución de métricas**: Accuracy y F1-score por ejecución
2. **Estabilización**: Medias acumuladas para analizar convergencia
3. **Z-scores**: Valores normalizados de las métricas
4. **Distribuciones**: Histogramas de frequency de las métricas

## 🚀 Cómo Ejecutar

### Prerrequisitos
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Ejecución
```bash
python arbolDecision.py
```

## 📋 Resultados Esperados

El programa muestra:

```
==================================================
ANÁLISIS DE CONSISTENCIA DEL MODELO
==================================================
Accuracy promedio: [valor]
Accuracy desviación estándar: [valor]
Accuracy rango: [min, max]

F1-score promedio: [valor]
F1-score desviación estándar: [valor]
F1-score rango: [min, max]

ANÁLISIS DE ESTABILIZACIÓN:
Diferencia entre últimas 10 y primeras 10 ejecuciones (Accuracy): [valor]
Diferencia entre últimas 10 y primeras 10 ejecuciones (F1): [valor]
==================================================
```

## 📚 Conceptos de Machine Learning Aplicados

- **Clasificación supervisada**
- **Árboles de decisión**
- **Validación cruzada temporal**
- **Métricas de evaluación**
- **Análisis de estabilidad de modelos**
- **Visualización de resultados**

## 🎓 Contexto Académico

Este proyecto forma parte del curso de **PROFUNDIZACIÓN I - MACHINE LEARNING** y demuestra:

- Implementación práctica de algoritmos de clasificación
- Evaluación rigurosa de modelos de ML
- Análisis estadístico de rendimiento
- Buenas prácticas en ciencia de datos

## Estructura del Proyecto

```
Arbol de decision/
├── arbolDecision.py                  # Script principal
├── dataset_sintetico_spam_ham.csv   # Dataset sintético
└── README.md                         # Documentación
```

## Descripción del Procedimiento y Algoritmo

Este script realiza una **clasificación binaria** sobre correos electrónicos utilizando **árboles de decisión** con análisis exhaustivo de consistencia y estabilidad del modelo.

**Procedimiento general:**

1. **Carga de datos:** Se utiliza un dataset sintético de 500 correos electrónicos con características específicas (longitud, adjuntos, enlaces, etc.) y metadatos asociados.

2. **Generación de etiquetas:** Las etiquetas SPAM/HAM se crean automáticamente según reglas específicas:
   - **SPAM (1)**: Si cumple alguna condición (remitente sospechoso, palabras de dinero, urgencia, o más de 2 enlaces)
   - **HAM (0)**: En caso contrario

3. **División de datos:** Para cada una de las 50 ejecuciones, los datos se dividen aleatoriamente en 70% entrenamiento y 30% prueba.

4. **Entrenamiento con Árbol de Decisión:** Se utiliza el criterio de Gini para la división de nodos. El algoritmo aprende patrones para distinguir entre correos legítimos y spam.

5. **Evaluación múltiple:** Se ejecuta el modelo 50 veces independientes para evaluar:
   - Consistencia de las métricas
   - Estabilidad del rendimiento
   - Variabilidad estadística

6. **Análisis estadístico:** Se calculan Z-scores, medias acumuladas, desviaciones estándar y rangos para ambas métricas (Accuracy y F1-score).

7. **Visualización completa:** Se generan 4 gráficos que muestran la evolución, estabilización, normalización y distribución de las métricas de rendimiento.

## 🎯 Objetivos del Análisis

- **Clasificación efectiva**: Distinguir automáticamente entre correos SPAM y HAM
- **Evaluación de robustez**: Medir la consistencia del modelo en múltiples ejecuciones
- **Análisis de estabilidad**: Verificar la convergencia de las métricas
- **Visualización del comportamiento**: Representar gráficamente el rendimiento del modelo

## Licencia

Uso académico.