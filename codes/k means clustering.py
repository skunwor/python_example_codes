# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:25:00 2020

@author: sujitk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline


plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12})
sns.set_style('whitegrid')
from sklearn.cluster import KMeans
wcss = []
for i in range(1,16):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1,16), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Clustes Sum of Squares')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10)
y_kmeans = kmeans.fit_predict(X_pca)

columns = np.append(boston.feature_names,['MEDV','PC 1', 'PC 2', 'Cluster'])
data = np.concatenate((boston.data, boston.target.reshape(-1,1), X_pca, y_kmeans.reshape(-1,1)), axis=1)
df_housing = pd.DataFrame(data=data, columns=columns)

df_housing.sample(5)

columns = np.append(boston.feature_names,['VARRATIO'])
data = np.concatenate((pca.components_, pca.explained_variance_ratio_.reshape(-1,1)), axis=1)
df_pca = pd.DataFrame(data=data, columns=columns, index=['PC 1', 'PC 2'])

df_pca

plt.figure(figsize=(10,8))
plt.xticks(rotation=45)
ax = sns.heatmap(df_pca.iloc[:,:-1], cmap='coolwarm', square=True, annot=True, cbar=False)

plt.figure(figsize=(10,8))
for i in range(kmeans.n_clusters):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], s = 75, label = 'Cluster ' + str(i+1))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s = 200, c = 'black', label = 'Centroids')
plt.title('Clusters of housing')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()


rows = 4
cols = 4
fif, axarr = plt.subplots(rows, cols, figsize=(15,15), sharey=True)
for i in range(rows):
    for j in range(cols):
        idx = i*rows + j
        if idx < len(boston.feature_names):
            axarr[i,j].scatter(boston.data[:,idx], boston.target, c=y_kmeans, cmap='Set1')
            axarr[i,j].set_title(boston.feature_names[idx])
            

#kde estimation plot by median price by cluster
plt.figure(figsize=(10,8))
for i in range(kmeans.n_clusters):
    #plt.hist(boston.target[y_kmeans == i], bins=20, label = 'Cluster ' + str(i+1), alpha=0.7)
    sns.kdeplot(boston.target[y_kmeans == i], shade=True, label='Cluster ' + str(i+1))
plt.title('Histogram of MEDV by cluster')
plt.xlabel('MEDV')
plt.legend()
plt.show()

