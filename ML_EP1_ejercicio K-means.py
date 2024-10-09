# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:15:40 2024

@author: Triton Perea

Machine Learning
Ejercicio del método k-means

Ejercicio 1: El dataset1 adjunto contiene 600 puntos en ℜ2. 
Encuentre los centros de los cinco grupos en los cuáles se distribuyen 
estos 600 puntos utilizando el método k-means. 
Grafique los puntos originales y los centros encontrados con k-means. 
 
Requerimiento: dataset1.csv -- es un archivo .csv que contiene una base de datos
con pares coordenados de puntos en un plano.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Valores iniciales del metodo, a ajustar según el usuario
delta = 1       
seed = 3
n_clusters = 5

# Lectura de datos desde el archivo CSV
ruta_csv= r"dataset1.csv"
data = pd.read_csv(ruta_csv, header=None)
dataset = np.array([[x, z] for x, z in zip(data[0], data[1])])
y= data.index.tolist()
n_samples = len(dataset)

# Inicializacion del metodo
zeros= np.zeros((n_samples,2))
gamma = np.zeros((n_samples, n_clusters))
n= np.zeros(n_clusters)
centroides = np.zeros((n_clusters,2))
nuevos_centroides = np.zeros((n_clusters,2))

# Graficar dataset original
plt.scatter(dataset[:,0], dataset[:,1], s=50)

# Selección random de los centroides
rdm = np.random.RandomState(seed)
a = rdm.permutation(dataset.shape[0])[:n_clusters]
centroides = dataset[a]

# Ciclo hasta la convergencia o 100 iteraciones
while True:
    
    """ Primer ciclo for del pseudocódigo
        Recorre punto por punto del dataset"""
    for i in range(n_samples):
        
        # Obtiene la distancia entre cada punto y los centroides
        distancias = np.linalg.norm(dataset[i] - centroides, axis=1)        
        
        # Asignación de centriode más cercano para cada punto
        cluster_cercano = np.argmin(distancias)
        gamma[i,:]=0                   # gamma_ij = 0 cualquier otra cosa
        gamma[i, cluster_cercano]=1    # gamma_ij = 1 si j = argmin(norma(punto-centroide))
        
    """ Segundo ciclo del pseudocódigo
        Recorre cada centroide para recalcular sus coordenadas"""
    for h in range(n_clusters):
        n[h] = np.sum(gamma[:, h])
        nuevos_centroides[h]= (1/n[h])*np.sum(gamma[:, h][:, np.newaxis] * dataset, axis=0)
        
    # Si hay convergencia, sale del ciclo
    if np.linalg.norm(nuevos_centroides-centroides) < 1e-3:
        break
    
    # Asigna valores correspondientes a nuevos centros
    centroides = nuevos_centroides
    
    # Limita las iteraciones del ciclo while ante una posible falta de convergencia
    delta-=0.01
    if delta < 0.1:
        break

print(centroides)

# Grafica de centroides con dataset original, modificar al gusto del usuario
plt.scatter(dataset[:, 0], dataset[:, 1], s=50, c='gray', label='Datos')
plt.scatter(centroides[:, 0], centroides[:, 1], s=20, c='red', label='Centroides Finales')
plt.title("Centroides Finales")
plt.legend()
plt.show()




