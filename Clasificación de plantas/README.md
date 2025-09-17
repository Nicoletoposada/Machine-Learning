# Clasificación de Plantas con Regresión Lineal (One-vs-Rest)

Este proyecto implementa un clasificador para el dataset IRIS utilizando regresión lineal bajo el esquema One-vs-Rest (OVR). Permite entrenar modelos, evaluar su precisión y realizar predicciones interactivas para nuevas plantas ingresando intervalos de características.

## Autores
- Maira Alejandra Chila
- Nicolás Posada García

## Descripción del Código
El archivo `iris_regresion_lineal.py` realiza las siguientes tareas:

1. **Carga del dataset IRIS** usando `scikit-learn`.
2. **Entrenamiento de modelos OVR:** Se entrena un modelo de regresión lineal para cada clase del dataset.
3. **Evaluación:** Se calcula la precisión y se muestra la matriz de confusión.
4. **Clasificación interactiva:** Permite al usuario ingresar intervalos para cada característica y predice la clase de todas las combinaciones posibles.

## Requisitos
- Python 3.x
- numpy
- scikit-learn

Puedes instalar las dependencias ejecutando:

```bash
pip install numpy scikit-learn
```

## Ejecución

Ejecuta el script desde la terminal:

```bash
python iris_regresion_lineal.py
```

Sigue las instrucciones en pantalla para ingresar los intervalos de características de una nueva planta.

## Estructura del Proyecto

```
Clasificación de plantas/
    iris_regresion_lineal.py
```


## Descripción del procedimiento y algoritmo

Este script realiza una **clasificación multiclase** sobre el dataset IRIS utilizando un enfoque de **regresión lineal One-vs-Rest (OVR)**. Además, permite al usuario ingresar intervalos para las características de una nueva planta y visualizar gráficamente la relación entre una característica y la probabilidad de pertenecer a una clase.

**Procedimiento general:**

1. **Carga de datos:** Se utiliza el dataset IRIS de `sklearn`, que contiene 150 muestras de flores con 4 características (largo y ancho de sépalo y pétalo) y 3 clases (tipos de iris).
2. **División de datos:** Los datos se dividen en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba) para evaluar el desempeño del modelo.
3. **Entrenamiento con One-vs-Rest:** Se entrena un modelo de regresión lineal para cada clase. Para cada modelo, la clase correspondiente se marca como 1 y las demás como 0 (binario). Así, cada modelo aprende a distinguir una clase contra el resto.
4. **Predicción y evaluación:** Para cada muestra de prueba, se obtienen las predicciones de todos los modelos y se asigna la clase con el mayor valor predicho. Se calcula la precisión y se muestra la matriz de confusión.
5. **Clasificación interactiva:** El usuario ingresa intervalos para cada característica de una nueva planta. El programa genera todas las combinaciones posibles dentro de esos intervalos y predice la clase para cada una.
6. **Visualización gráfica:** Al final, se muestra una gráfica de dispersión de la primera característica (por ejemplo, largo del sépalo) contra la probabilidad de pertenecer a la primera clase, usando la regresión lineal entrenada. Las otras características se mantienen en su valor medio para la predicción.


## Licencia
Uso académico.