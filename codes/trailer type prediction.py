# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:26:03 2020

@author: sujitk
"""


import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from keras import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=DataScience01;"
                      "Database=DS01;"
                      "Trusted_Connection=yes;")
cursor = conn.cursor()


df = pd.read_sql(
 """SELECT 
         dest_market_id
         ,commodity
        ,origin_market_id
        ,freight_chg
        ,pay_distance
        ,origin_yrweek
        ,spot_freight_pm
        ,fsc
        ,spot_fsc
        ,spot_other_chg
        ,other_chg
        ,spot_tot_chgs
        ,tot_chgs
        ,spot_freight_chg
        ,freight_pm
        ,exchange_type
        ,dest_state
        ,origin_state
    FROM [DS01].[dbo].[x_rate_history] 
    where exclude_from_rate = 'N' and exchange_type != 'O' and origin_actual_av between '01-01-2019' and '10-01-2019'
    
    union 
    
    SELECT 
         dest_market_id
         ,commodity
        ,origin_market_id
        ,freight_chg
        ,pay_distance
        ,origin_yrweek
        ,spot_freight_pm
        ,fsc
        ,spot_fsc
        ,spot_other_chg
        ,other_chg
        ,spot_tot_chgs
        ,tot_chgs
        ,spot_freight_chg
        ,freight_pm
        ,exchange_type
        ,dest_state
        ,origin_state
    FROM [DS01].[dbo].[vw_x_rate_history_YTD] 
    where exclude_from_rate = 'N' and exchange_type != 'O' and origin_actual_av between '01-01-2020' and '10-01-2020'
    """,conn)
    
    

df.commodity.fillna(value = 'unknown',inplace = True)
df = df.dropna()
y = df[['exchange_type']].values
X = df.loc[:,df.columns != 'exchange_type'].values


encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 15] = encoder.fit_transform(X[:, 15])
X[:, 16] = encoder.fit_transform(X[:, 16])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1007)


X_train=np.array(X_train).astype(np.float32)
y_train= encoder.fit_transform(y_train)
y_train=np.array(y_train).astype(np.float32)

y_test= encoder.fit_transform(y_test)
y_test=np.array(y_test).astype(np.float32)

# clf_knn2 = KNeighborsClassifier(n_neighbors = 7)
# clf_knn2.fit(X_train,y_train)
# confusion_matrix(y_test, clf_knn2.predict(X_test))
# accuracy_score(y_test, clf_knn2.predict(X_test))

# clf_knn5 = KNeighborsClassifier(n_neighbors = 5)
# clf_knn5.fit(X_train_s, y_train)
# accuracy_score(y_test, clf_knn5.predict(X_test_s))
# confusion_matrix(y_test, clf_knn5.predict(X_test_s))




#deep learning 

model = Sequential([
    Dense(100, activation='relu', input_shape = (X_train.shape[1],)  ),
    Dense(100, activation='softmax'),
    Dense(3, activation='softmax'),
])
opt = SGD(lr = 0.1)

from tensorflow import keras

#opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',metrics = ['accuracy']
              )
#ex_fit = model.fit(X_train, dummy_y, epochs=10)

ex_fit = model.fit(X_train, y_train, batch_size = 500000, epochs=50)
























