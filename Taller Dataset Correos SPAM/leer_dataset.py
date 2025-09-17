import pandas as pd

# Cargar el dataset
df = pd.read_csv('dataset_sintetico_spam_ham.csv')

# Mostrar las primeras filas

def mostrar_info():
    print('Primeras filas del dataset:')
    print(df.head())
    print('\nInformación general:')
    print(df.info())
    print('\nDistribución de clases (calculada por reglas):')
    # Clasificación por reglas
    condiciones_spam = (df['remitente_sospechoso'] == 1) & (df['contiene_palabras_dinero'] == 1)
    condiciones_ham = (df['remitente_sospechoso'] == 0) & (df['contiene_palabras_dinero'] == 0) & (df['urgencia'] == 0)
    # Por defecto, si no cumple ninguna, lo marcamos como 'Desconocido'
    clases = ['Desconocido'] * len(df)
    for i in df.index:
        if condiciones_spam[i]:
            clases[i] = 'SPAM'
        elif condiciones_ham[i]:
            clases[i] = 'HAM'
    df['clase'] = clases
    print(df['clase'].value_counts())

if __name__ == '__main__':
    mostrar_info()
