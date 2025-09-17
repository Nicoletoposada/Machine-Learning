
# Clasificación de Correos HAM/SPAM con Regresión Logística

Este proyecto implementa un modelo de **Machine Learning** para
clasificar correos electrónicos como **HAM (legítimos)** o **SPAM**
usando **Regresión Logística** en **Python** con la librería
`scikit-learn`.

## Dataset


El dataset utilizado se encuentra en el archivo `dataset_sintetico_spam_ham.csv` y está basado en la descripción del documento **Taller Dataset Correos SPAM (PDF)**.
Incluye las siguientes variables:

-   **nombre**: Nombre del remitente del correo.
-   **email**: Dirección de correo del remitente.
-   **fecha**: Fecha de envío.
-   **asunto**: Texto del asunto del correo.
-   **longitud_cuerpo**: Número de caracteres del cuerpo del correo.
-   **num_adjuntos**: Cantidad de archivos adjuntos.
-   **num_links**: Número de enlaces dentro del correo.
-   **remitente_sospechoso**: (0/1) Indica si el remitente es
    sospechoso.
-   **contiene_palabras_dinero**: (0/1) Indica si contiene palabras
    asociadas a dinero/premios.
-   **urgencia**: (0/1) Indica si el correo contiene mensajes de
    urgencia.

-   **clase**: Variable objetivo → `0 = HAM`, `1 = SPAM`.
    - Esta columna se genera automáticamente en el script, siguiendo la siguiente lógica:
        - Un correo es **SPAM** (`clase=1`) si cumple **todas** las siguientes condiciones:
            - `remitente_sospechoso == 1`
            - `contiene_palabras_dinero == 1`
            - `urgencia == 1`
        - En caso contrario, se considera **HAM** (`clase=0`).

## Implementación


El modelo está implementado en el script `machineRegresion.py` y sigue el siguiente flujo:

1.  **Carga del dataset** desde el archivo CSV.
2.  **Generación de la variable objetivo `clase`** según reglas lógicas sobre las variables del dataset.
3.  **Selección de variables predictoras**: se usan únicamente variables numéricas y binarias relevantes para la clasificación (`longitud_cuerpo`, `num_adjuntos`, `num_links`, `remitente_sospechoso`, `contiene_palabras_dinero`, `urgencia`).
4.  **División de datos** en conjuntos de entrenamiento y prueba (80%/20%) usando muestreo estratificado.
5.  **Normalización de variables** con `StandardScaler` para mejorar el desempeño del modelo.
6.  **Entrenamiento con Regresión Logística** usando búsqueda de hiperparámetros (`GridSearchCV`):
    -   Parámetro `C`: controla la regularización.
    -   Penalización: `l1` y `l2`.
    -   Solver: `saga` (compatible con ambas penalizaciones).
    -   Validación cruzada con `cv=3` (adecuado para datasets pequeños).
7.  **Evaluación del modelo** con métricas de clasificación:
    -   Se usa **F1-Score** como métrica principal.
    -   Se imprime el reporte de clasificación completo.
8.  **Ecuación final del modelo**: se muestran el intercepto y los coeficientes de cada variable, permitiendo interpretar la contribución de cada predictor.

## Métricas


El modelo devuelve en consola:


-   **Mejores hiperparámetros encontrados** con `GridSearchCV`.
-   **F1-Score en el conjunto de prueba**.
-   **Reporte de clasificación** (precision, recall, f1-score, support).
-   **Ecuación de regresión logística** en la forma:


\[ Logit(P) = \beta_0 + \beta_1 \cdot longitud\_cuerpo + \beta_2 \cdot num\_adjuntos + ... + \beta_n \cdot urgencia \]

Donde:

\[ P(SPAM) = \frac{1}{1 + e^{-Logit(P)}} \]

## Ejecución


### 1. Instalar dependencias:

``` bash
pip install scikit-learn pandas numpy
```


### 2. Ejecutar el script:

``` bash
python machineRegresion.py
```


### 3. Revisar la salida en consola:
    -   Hiperparámetros óptimos encontrados.
    -   F1-Score en test.
    -   Reporte de clasificación.
    -   Ecuación de regresión logística.

## Notas


-   Si el dataset es pequeño, se recomienda usar menos `cv` en `GridSearchCV` (por ejemplo `cv=3`).
-   Para mejorar resultados, usar un dataset con más registros HAM y SPAM balanceados.
-   El script está preparado para trabajar principalmente con archivos **CSV**.
-   La lógica de clasificación puede ajustarse según las necesidades del problema.

------------------------------------------------------------------------


---

Autor: *Maira Alejandra Chila - Nicolás Posada García*
