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

## Notas
- El script es interactivo y requiere que el usuario ingrese valores por consola.
- La clasificación de nuevas plantas se realiza generando todas las combinaciones posibles dentro de los intervalos proporcionados.

## Licencia
Uso académico.