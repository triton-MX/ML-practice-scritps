# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:59:51 2024


@author: Triton Perea

Machine Learning
Eejercicio Maquina de Soporte Vectorial suave (SVC)

Ejercicio 4:
Tome los puntos de las clases con etiquetas 0 y 1 del dataset3_train. 
Utilice estos puntos para entrenar un clasificador por Máquina de Soporte Vectorial 
de margen suave (SVC). Utilice un kernel linear y como hiperparámetro un valor de C=10. 
Posteriormente, utilice los puntos del dataset3_test que tienen las etiquetas 
de clase 0 y 1 para evaluar el clasificador entrenado. Genere una gráfica mostrando 
las fronteras de decisión del clasificador. Muestre la matriz de confusión asociada
a la evaluación de su clasificador y el reporte de clasificación que se obtiene con
los puntos de prueba. .
"""
#Librerias a usar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Rutas de dataset en csv
ruta_csv_train= r"C:\...\dataset3_train.csv"
ruta_csv_test = r"C:\...\dataset3_test.csv"

# Valores iniciales del metodo
kernel='linear'
C=10

# Importar datos de archivo CSV
def importarCSV(ruta_csv):
    data = pd.read_csv(ruta_csv, header=None)
    filter_data = data[data[2].isin([0, 1])]    # Solo tomamos los puntos con etiqueta 0 o 1
    dataset = np.array([[x, y] for x, y in zip(filter_data[0], filter_data[1])])
    z=  filter_data[2].values       # Solo tomamos las etiquetas 0 o 1
    return dataset, z

# Dataset de entrenamiento
training_data, training_labels= importarCSV(ruta_csv_train)

# Dataset de prueba
test_data, test_labels= importarCSV(ruta_csv_test)

# Uso de metodo SVC de la libreria sklearn
modelo= SVC(kernel=kernel, C=C)
modelo.fit(training_data, training_labels)

# Grafica de fronteras de decision
x_min, x_max = training_data[:, 0].min() - 1, training_data[:, 0].max() + 1
y_min, y_max = training_data[:, 1].min() - 1, training_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.scatter(training_data[:, 0], training_data[:, 1], c=training_labels, edgecolor='k', s=50)
plt.title('Fronteras de Decisión del SVC')
plt.axis('equal')
plt.grid()
plt.show()

# Predicciones del SVC don datos de prueba
pred_labels= modelo.predict(test_data)

# Evaluacion de matriz de confusion y reporte de clasificacion
matriz = confusion_matrix(test_labels, pred_labels)
reporte = classification_report(test_labels, pred_labels)
print(f"Matriz de confusion:\n {matriz}")
print(f"Reporte de clasificacion:\n {reporte}")

