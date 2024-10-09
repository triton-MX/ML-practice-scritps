# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:44:39 2024

@author: Triton Perea

Machine Learning
Ejercicio PCA

Ejercicio 3: 
Utilice el método PCA para encontrar las primeras 2 componentes principales de 
los datos proporcionados en el dataset2. Grafique un scatter plot con los datos
originales proyectados sobre las dos componentes principales encontradas. 

Requerimientos: dataset2.csv -- archivo csv con vectores de 4 dimensiones
"""
# Importar liberias a usar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Valores iniciales del metodo
n_componentes=2

# Lectura de datos desde el archivo CSV
ruta_csv= r"C:\...\dataset2.csv"
data = pd.read_csv(ruta_csv, header=None)
dataset = np.array([[w,x,y,z] for w,x,y,z in zip(data[0], data[1], data[2], data[3])])

# Uso de PCA para 2 primeras componentes principales, con libreria sklearn
pca = PCA(n_components=n_componentes)
data_transform= pca.fit_transform(dataset)

# Grafica de resultados
plt.scatter(data_transform[:, 0], data_transform[:, 1], cmap='viridis', edgecolor='k')
plt.title('PCA')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.axis('equal')
plt.grid()
plt.show()
