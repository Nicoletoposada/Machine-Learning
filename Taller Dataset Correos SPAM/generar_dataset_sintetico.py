import pandas as pd
import numpy as np
from faker import Faker
import random

faker = Faker('es_ES')

# Palabras y asuntos típicos de SPAM y HAM
asuntos_spam = [
    '¡Gana dinero rápido!', 'Premio en efectivo', '¡Has sido seleccionado!', 'Oferta exclusiva para ti',
    '¡Última oportunidad!', 'Recibe tu dinero', 'Transferencia pendiente', '¡Gana un iPhone!',
    'Dinero fácil y rápido', '¡Responde ahora!', '¡Gana un viaje!', 'Premio asegurado solo hoy!'
]
asuntos_ham = [
    'Reunión semanal', 'Informe mensual', 'Cita médica', 'Agenda de actividades',
    'Minuta de reunión', 'Reporte de ventas', 'Plan de trabajo', 'Lista de asistencia',
    'Recordatorio de pago', 'Agenda semanal', 'Informe trimestral', 'Plan de actividades'
]


# Genera 1000 correos, mitad SPAM y mitad HAM, pero sin columna label.
n = 1000
labels = np.random.choice(['SPAM', 'HAM'], size=n, p=[0.5,0.5])
data = []

for label in labels:
    if label == 'SPAM':
        nombre = faker.name()
        email = faker.email()
        fecha = faker.date_between(start_date='-2y', end_date='today')
        asunto = random.choice(asuntos_spam)
        longitud_cuerpo = np.random.randint(800, 1400)
        num_adjuntos = np.random.choice([0,1,2], p=[0.3,0.4,0.3])
        num_links = np.random.randint(2, 6)
        remitente_sospechoso = 1
        contiene_palabras_dinero = 1
        urgencia = np.random.choice([0,1], p=[0.3,0.7])
    else:
        nombre = faker.name()
        email = faker.email()
        fecha = faker.date_between(start_date='-2y', end_date='today')
        asunto = random.choice(asuntos_ham)
        longitud_cuerpo = np.random.randint(800, 1400)
        num_adjuntos = np.random.choice([0,1], p=[0.8,0.2])
        num_links = np.random.randint(0, 2)
        remitente_sospechoso = 0
        contiene_palabras_dinero = 0
        urgencia = 0
    # No se agrega la columna label
    data.append([
        nombre, email, fecha, asunto, longitud_cuerpo, num_adjuntos, num_links,
        remitente_sospechoso, contiene_palabras_dinero, urgencia
    ])


# Columnas del dataset (sin label)
columns = [
    'nombre','email','fecha','asunto','longitud_cuerpo','num_adjuntos','num_links',
    'remitente_sospechoso','contiene_palabras_dinero','urgencia'
]
df = pd.DataFrame(data, columns=columns)
df.to_csv('dataset_sintetico_spam_ham.csv', index=False)

# ¿Cómo identificar SPAM y HAM?
# Un correo es SPAM si:
#   remitente_sospechoso == 1 y contiene_palabras_dinero == 1
#   (urgencia suele ser 1, pero puede ser 0)
# Un correo es HAM si:
#   remitente_sospechoso == 0 y contiene_palabras_dinero == 0 y urgencia == 0
# Puedes usar estas reglas en tu código para clasificar los correos.

print('Dataset sintético generado y guardado como dataset_sintetico_spam_ham.csv (sin columna label)')
