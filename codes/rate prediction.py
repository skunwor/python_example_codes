# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:13:54 2020

@author: sujitk
"""
import pyodbc
import pandas as pd
conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=DataScience01;"
                      "Database=DS01;"
                      "Trusted_Connection=yes;")
cursor = conn.cursor()


df = pd.read_sql(
 """SELECT top (20000)
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
    where exclude_from_rate = 'N'
    
    union 
    
    SELECT top (20000)
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
    where exclude_from_rate = 'N'
    """,conn)
    
    



from sklearn.model_selection import train_test_split
df.commodity.fillna(value = 'unknown',inplace = True)


y = df[['freight_chg']].values
X = df.loc[:,df.columns != 'freight_chg'].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])

X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 15] = encoder.fit_transform(X[:, 15])
X[:, 14] = encoder.fit_transform(X[:, 14])
X[:, 16] = encoder.fit_transform(X[:, 16])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1007)
import numpy as np
from keras import utils

X_train=np.array(X_train).astype(np.float32)
y_train= encoder.fit_transform(y_train)
y_train=np.array(y_train).astype(np.float32)


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD



model = Sequential([
    Dense(50, activation='softmax'),
    Dense(4, activation='softmax'),
])
opt = SGD(lr = 0.01, momentum = 0.9)
model.compile(optimizer=opt,
              loss='mse',metrics = ['accuracy']
              )
#ex_fit = model.fit(X_train, dummy_y, epochs=10)

ex_fit = model.fit(X_train, y_train, epochs=10)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



clf_knn2 = KNeighborsClassifier(n_neighbors = 500)
clf_knn2.fit(X_train,y_train)
confusion_matrix(y_test, clf_knn2.predict(X_test_s))
accuracy_score(y_test, clf_knn2.predict(X_test_s))



























