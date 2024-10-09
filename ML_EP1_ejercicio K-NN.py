# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:05:59 2024
@author: Triton Perea

Machine Learning
Ejercicio K-NN

Ejercicio 2: El dataset3_train contiene 1000 vectores etiquetados. 
Las dos primeras columnas son las componentes de los vectores y la tercera columna 
las etiquetas correspondientes a 1 de 3 clases. Utilice estos 1000 puntos para 
construir un clasificador k-NN con k = 5. Posteriormente, utilice los 200 puntos 
del dataset3_test para evaluar el clasificador entrenado. 
Genere una gráfica mostrando las fronteras de decisión del clasificador. 
Muestre la matriz de confusión asociada a la evaluación de su clasificador y 
el reporte de clasificación que se obtiene con los 200 puntos de prueba. 

"""
#Se importan librerias a usar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Rutas de dataset en csv
ruta_csv_train= r"C:\...\dataset3_train.csv"
ruta_csv_test = r"C:\...\dataset3_test.csv"

# Valores iniciales del clasificador
k=5

# Importar datos de archivo CSV
def importarCSV(ruta_csv):
    data = pd.read_csv(ruta_csv, header=None)
    dataset = np.array([[x, y] for x, y in zip(data[0], data[1])])
    z= np.array(data[2])
    return dataset, z

# Dataset de entrenamiento
training_data, training_labels= importarCSV(ruta_csv_train)

# Dataset de prueba
test_data, test_labels= importarCSV(ruta_csv_test)

# Instacia clasificadora del KNN
knn= KNeighborsClassifier(n_neighbors=k)

# Ajuste del modelo
knn.fit(training_data,training_labels)

# Predicciones en el conjunto prueba
pred = knn.predict(test_data)

# Evaluar la matriz de confusion y reporte
matrizC = confusion_matrix(test_labels, pred)
report = classification_report(test_labels, pred)
print(f"Matriz de confusion:\n: {matrizC}")
print(f"Reporte:\n {report}")

# Uso de sklearn para obtener las fronteras de decision
x_min, x_max = training_data[:, 0].min() - 1, training_data[:, 0].max() + 1
y_min, y_max = training_data[:, 1].min() - 1, training_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# El listado de colores depende de la cantidad de clases que se tenga
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

# Ajustar el tamaño de la gráfica
plt.figure(figsize=(8, 6))

# Graficar las fronteras de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Graficar los puntos de entrenamiento
plt.scatter(training_data[:, 0], training_data[:, 1], c=training_labels, cmap=cmap_bold, edgecolor='k', s=20)

# Graficar los puntos de prueba con predicciones
plt.scatter(test_data[:, 0], test_data[:, 1], c=pred, cmap=cmap_bold, edgecolor='k', marker='x', s=50)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"Fronteras de Decisión KNN (k={k})")
plt.xlabel("C 1")
plt.ylabel("C 2")
plt.show
