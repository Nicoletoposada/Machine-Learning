import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# ==========================
# 1. Cargar dataset
# ==========================
df = pd.read_csv("dataset_sintetico_spam_ham.csv")


# ==========================
# 2. Crear la columna label (SPAM=1, HAM=0)
# ==========================
df["label"] = ((df["remitente_sospechoso"] == 1) |
               (df["contiene_palabras_dinero"] == 1) |
               (df["urgencia"] == 1) |
               (df["num_links"] > 2)).astype(int)

print("Distribución de clases:\n", df["label"].value_counts())

# ==========================
# 3. Preparar variables
# ==========================
X = df.drop(columns=["nombre", "email", "fecha", "asunto", "label"])
y = df["label"]

accuracies = []
f1_scores = []

# ==========================
# 4. Ejecutar 50 veces el modelo
# ==========================
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")  # Árbol de decisión simple
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

accuracies = np.array(accuracies)
f1_scores = np.array(f1_scores)

# ==========================
# 5. Calcular Z-score
# ==========================
def z_score(arr):
    if arr.std() == 0:
        return np.zeros_like(arr)  # evitar división por cero
    return (arr - arr.mean()) / arr.std()

z_acc = z_score(accuracies)
z_f1 = z_score(f1_scores)

# ==========================
# 6. Graficar resultados
# ==========================
# Calcular medias acumuladas para analizar estabilización
running_mean_acc = np.cumsum(accuracies) / np.arange(1, len(accuracies) + 1)
running_mean_f1 = np.cumsum(f1_scores) / np.arange(1, len(f1_scores) + 1)

plt.figure(figsize=(15,10))

# Accuracy y F1 por ejecución
plt.subplot(2,2,1)
plt.plot(range(1,51), accuracies, marker="o", alpha=0.7, label="Accuracy")
plt.plot(range(1,51), f1_scores, marker="x", alpha=0.7, label="F1-score")
plt.xlabel("Ejecución")
plt.ylabel("Valor")
plt.title("Evolución de Accuracy y F1-score en 50 ejecuciones")
plt.legend()
plt.grid(True, alpha=0.3)

# Medias acumuladas para analizar estabilización
plt.subplot(2,2,2)
plt.plot(range(1,51), running_mean_acc, marker="o", label="Media acumulada Accuracy")
plt.plot(range(1,51), running_mean_f1, marker="x", label="Media acumulada F1-score")
plt.xlabel("Ejecución")
plt.ylabel("Media acumulada")
plt.title("Estabilización de las métricas (medias acumuladas)")
plt.legend()
plt.grid(True, alpha=0.3)

# Z-scores
plt.subplot(2,2,3)
plt.plot(range(1,51), z_acc, marker="o", label="Z-score Accuracy")
plt.plot(range(1,51), z_f1, marker="x", label="Z-score F1")
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel("Ejecución")
plt.ylabel("Z-score")
plt.title("Z-scores de las métricas")
plt.legend()
plt.grid(True, alpha=0.3)

# Histogramas de distribución
plt.subplot(2,2,4)
plt.hist(accuracies, alpha=0.7, bins=15, label="Accuracy", color='blue')
plt.hist(f1_scores, alpha=0.7, bins=15, label="F1-score", color='orange')
plt.xlabel("Valor de la métrica")
plt.ylabel("Frecuencia")
plt.title("Distribución de las métricas")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================
# 7. Resultados finales
# ==========================
print("="*50)
print("ANÁLISIS DE CONSISTENCIA DEL MODELO")
print("="*50)
print(f"Accuracy promedio: {accuracies.mean():.4f}")
print(f"Accuracy desviación estándar: {accuracies.std():.4f}")
print(f"Accuracy rango: [{accuracies.min():.4f}, {accuracies.max():.4f}]")
print()
print(f"F1-score promedio: {f1_scores.mean():.4f}")
print(f"F1-score desviación estándar: {f1_scores.std():.4f}")
print(f"F1-score rango: [{f1_scores.min():.4f}, {f1_scores.max():.4f}]")
print()
print("ANÁLISIS DE ESTABILIZACIÓN:")
print(f"Diferencia entre últimas 10 y primeras 10 ejecuciones (Accuracy): {running_mean_acc[-10:].mean() - running_mean_acc[:10].mean():.4f}")
print(f"Diferencia entre últimas 10 y primeras 10 ejecuciones (F1): {running_mean_f1[-10:].mean() - running_mean_f1[:10].mean():.4f}")
print("="*50)
