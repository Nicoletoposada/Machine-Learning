import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# ================================
# 1. Leer dataset desde archivo CSV
# ================================
df = pd.read_csv("dataset_sintetico_spam_ham.csv")

# Crear la columna 'clase': 1 si es SPAM, 0 si es HAM
# Suponemos que un correo es SPAM si contiene palabras de dinero, urgencia y remitente sospechoso (como en los datos sintéticos)
df["clase"] = ((df["remitente_sospechoso"] == 1) & (df["contiene_palabras_dinero"] == 1) & (df["urgencia"] == 1)).astype(int)

print("✅ Dataset cargado desde CSV")
print(df[["longitud_cuerpo","num_adjuntos","num_links","remitente_sospechoso","contiene_palabras_dinero","urgencia","clase"]].head())

# ================================
# 2. Variables predictoras y target
# ================================
X = df[["longitud_cuerpo","num_adjuntos","num_links",
        "remitente_sospechoso","contiene_palabras_dinero","urgencia"]]

y = df["clase"]

# ================================
# 3. Dividir en train/test
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado de variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 4. Regresión logística con GridSearchCV (variabilidad)
# ================================
logreg = LogisticRegression(max_iter=1000)

param_grid = {
    "C": [0.01, 0.1, 1, 10],    # Regularización
    "penalty": ["l1", "l2"],    # Tipos de penalización
    "solver": ["saga"]          # Compatible con l1 y l2
}

#Como el dataset es pequeño, usamos cv=3
grid = GridSearchCV(
    logreg,
    param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

# ================================
# 5. Evaluación
# ================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\n================ RESULTADOS ================")
print("Mejores hiperparámetros:", grid.best_params_)
print("F1-Score en test:", f1_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# ================================
# 6. Ecuación de la regresión logística
# ================================
coeficientes = best_model.coef_[0]
intercepto = best_model.intercept_[0]
variables = X.columns

ecuacion = f"Logit(P) = {intercepto:.4f} "
for var, coef in zip(variables, coeficientes):
    ecuacion += f"+ ({coef:.4f} * {var}) "

print("\nEcuación de Regresión Logística (mejor modelo):")
print(ecuacion)
print("\nProbabilidad SPAM = 1 / (1 + exp(-Logit(P)))")
