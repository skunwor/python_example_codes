# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:09:52 2020

@author: sujitk
"""
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('C:/Users/sujitk/OneDrive - Tom McLeod Software, Inc/Movie_classification.csv', header = 0)

df.head()

df.info()
df.shape()

df.describe()

sns.boxplot(y = 'Movie_length',data = df)

sns.scatter(y = 'Marketing expense', x = 'Production expense')

sns.pairplot(data = df,height = 10,y_vars = 'Production expense', x_vars = 'Marketing expense')


df = pd.get_dummies(df)
del df['3D_available_NO']
del df['Genre_Action']

X = df.loc[:,df.columns != 'Start_Tech_Oscar']
X = X.loc[:,X.columns != 'Time_taken']

y = df[['Start_Tech_Oscar']]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s= scaler.transform(X_train)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_s= scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

clf_knn2 = KNeighborsClassifier(n_neighbors = 2)
clf_knn2.fit(X_train_s,y_train)
confusion_matrix(y_test, clf_knn2.predict(X_test_s))
accuracy_score(y_test, clf_knn2.predict(X_test_s))

clf_knn5 = KNeighborsClassifier(n_neighbors = 5)
clf_knn5.fit(X_train_s, y_train)
accuracy_score(y_test, clf_knn5.predict(X_test_s))
confusion_matrix(y_test, clf_knn5.predict(X_test_s))

































