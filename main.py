# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:55:30 2022

@author: David
"""
import numpy as np
import pandas as pd
from functions import bordersToFeatures, openImages, gaussian_kernel, applyConvolve, sobel_filters, non_max_suppression, threshold, hysteresis, saveImages

# %% PREPROCESAMIENTO
letters = ['A', 'B', 'C', 'D', 'E']
for letter in letters:
    imgs=openImages(f'Referencia {letter}') #Carga de imagenes
    #Find Edges
    imgsC=applyConvolve(imgs, gaussian_kernel())
    imgsS, Ds=sobel_filters(imgsC)
    results, weaks, strongs = threshold(imgsS)
    saveImages(results, f'Bordes{letter}', f'{letter}') #Guardan la imagen con bordes resaltados

# %% DETERMINACION CONTORNOS Y SELECCION DE CARACTERISTICAS
toWork = {}
folderNames = ['BordesA','BordesB','BordesC','BordesD','BordesE']
for name in folderNames:
    A = bordersToFeatures(name) #Retorna características
    toWork[name] = A
# %% CREACION DATAFRAME CON EL ESPACIO DE CARACTERISTICAS PARA EVALUAR MODELOS
label = np.array([])
values = []
values2 = []
values3 = []
for i, name in enumerate(folderNames):
    newlabels = np.zeros(len(list(toWork[name].values()))) + i
    label = np.concatenate((label, newlabels))
    values += [row[0] for row in list(toWork[name].values())]
    values2 += [row[1] for row in list(toWork[name].values())]
    values3 += [row[2] for row in list(toWork[name].values())]
DF = pd.DataFrame({'Area': values, 'Ratio': values2, 'Longitud': values3, 'label': label})

# %% BUSQUEDA DE PARAMETROS SVM

from sklearn.model_selection import GridSearchCV
from sklearn import svm

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
clf = GridSearchCV(svm.SVC(), parameters, cv= 4)
clf.fit(DF[DF.columns[:-1]], DF.label)
print('PRECISION PARAMETROS SVM')
print('kernel: linear/C: 1, kernel: linear/C: 10, kernel: linear/C: 100')
print(clf.cv_results_['mean_test_score'][:3])
print('kernel: rbf/C: 1, kernel: rbf/C: 10, kernel: rbf/C: 100')
print(clf.cv_results_['mean_test_score'][3:])
# %% BUSQUEDA DE PARAMETROS KNN
print('')
from sklearn.neighbors import KNeighborsClassifier

parameters = {'weights':['uniform', 'distance'],
              'metric':['manhattan']}
clf = GridSearchCV(KNeighborsClassifier(n_neighbors=3), parameters, cv= 3)
clf.fit(DF[DF.columns[:-1]], DF.label)
print('PRECISION PARAMETROS KNN')
print('weights: uniform, distance')
print(clf.cv_results_['mean_test_score'])

# %% SEPARACION CONJUNTO DE ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DF[DF.columns[:-1]], DF.label, stratify=DF.label, random_state=0);
# %% IMPLEMENTACION MODELO KNN CON PARAMETROS OPTIMOS
print('')
print('K-VECINOS MAS CERCANOS')
Knn=KNeighborsClassifier(n_neighbors = 3, weights='distance', metric='manhattan');
Knn.fit(X_train, y_train);
y_predict=Knn.predict(X_test)
print('PRECISION DATOS ENTRENAMIENTO')
print(Knn.score(X_train, y_train))
print('PRECISION DATOS PRUEBA')
print(Knn.score(X_test, y_test))

# %% IMPLEMENTACION MODELO SVM CON PARAMETROS OPTIMOS
print('')
print('MAQUINA DE SOPORTE VECTORIAL')
Svc=svm.SVC(kernel = 'rbf', C=100);
Svc.fit(X_train, y_train);
print('PRECISION DATOS ENTRENAMIENTO')
print(Svc.score(X_train, y_train))
print('PRECISION DATOS PRUEBA')
print(Svc.score(X_test, y_test))
# %% IMPLEMENTACION MODELO GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier
print('')
print('GRADIENT BOOSTING')
clf=GradientBoostingClassifier()
clf.fit(X_train, y_train)
y_pred_proba=clf.fit(X_train, y_train).predict_proba(X_test)
print('PRECISION DATOS ENTRENAMIENTO')
print(clf.score(X_train, y_train))
print('PRECISION DATOS PRUEBA')
print(clf.score(X_test, y_test))