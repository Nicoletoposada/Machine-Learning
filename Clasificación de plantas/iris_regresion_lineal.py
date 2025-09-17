"""
Autor: Maira Alejandra Chila y Nicolás Posada García
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets


def cargar_datos():
    """Carga el dataset IRIS."""
    data = datasets.load_iris()
    return data.data, data.target, data.target_names, data.feature_names

def entrenar_modelos_ovr(X_train, y_train):
    """Entrena un modelo de regresión lineal para cada clase (One-vs-Rest)."""
    modelos = []
    for clase in np.unique(y_train):
        y_bin = (y_train == clase).astype(int)
        modelo = LinearRegression().fit(X_train, y_bin)
        modelos.append(modelo)
    return modelos

def predecir_ovr(modelos, X):
    """Predice la clase para cada muestra usando los modelos OVR."""
    predicciones = []
    for fila in X:
        scores = [modelo.predict([fila])[0] for modelo in modelos]
        predicciones.append(np.argmax(scores))
    return np.array(predicciones)

def mostrar_matriz_confusion_texto(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión:")
    print("\t" + "\t".join(target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]}\t" + "\t".join(str(x) for x in row))

def clasificar_nueva_planta_intervalo(modelos, target_names, feature_names, X):
    print("\nRangos válidos de las características del dataset IRIS:")
    for i, nombre in enumerate(feature_names):
        minimo = np.min(X[:, i])
        maximo = np.max(X[:, i])
        print(f"- {nombre}: mínimo = {minimo:.2f}, máximo = {maximo:.2f}")
    print("\nIntroduce los intervalos de las características de la nueva planta.")
    print("Para cada característica, ingresa el valor mínimo y máximo separados por un espacio (ejemplo: 4.5 5.5)")
    rangos = []
    for i, nombre in enumerate(feature_names):
        min_val = np.min(X[:, i])
        max_val = np.max(X[:, i])
        while True:
            entrada = input(f"{nombre} (min {min_val:.2f} max {max_val:.2f}): ")
            partes = entrada.strip().split()
            if len(partes) != 2:
                print("Debes ingresar dos valores: mínimo y máximo.")
                continue
            try:
                minimo, maximo = float(partes[0]), float(partes[1])
            except ValueError:
                print("Debes ingresar valores numéricos.")
                continue
            if minimo > maximo:
                print("El mínimo no puede ser mayor que el máximo.")
                continue
            if minimo < min_val or maximo > max_val:
                print(f"El intervalo debe estar entre {min_val:.2f} y {max_val:.2f}.")
                continue
            rangos.append(np.linspace(minimo, maximo, num=3))  # 3 valores por característica
            break

    # Generar todas las combinaciones posibles
    from itertools import product
    combinaciones = list(product(*rangos))
    print(f"\nClasificando {len(combinaciones)} combinaciones posibles:")
    for idx, valores in enumerate(combinaciones, 1):
        nueva_planta = np.array([valores])
        scores = [modelo.predict(nueva_planta)[0] for modelo in modelos]
        clase_predicha = np.argmax(scores)
        print(f"Combinación {idx}: {valores} => {target_names[clase_predicha]}")



def main():
    # 1. Cargar datos
    X, y, target_names, feature_names = cargar_datos()

    # 2. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Entrenar modelos OVR
    modelos = entrenar_modelos_ovr(X_train, y_train)

    # 4. Predecir y evaluar
    y_pred = predecir_ovr(modelos, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo (regresión lineal One-vs-Rest): {accuracy:.2f}")

    # 5. Matriz de confusión (solo texto)
    mostrar_matriz_confusion_texto(y_test, y_pred, target_names)

    # 6. Clasificación interactiva con intervalos
    clasificar_nueva_planta_intervalo(modelos, target_names, feature_names, X)

if __name__ == "__main__":
    main()
