# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:59:27 2020

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
 """SELECT top (200000)
 scac,
         dest_market_id
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
    
    SELECT top (200000)
    scac,
         dest_market_id
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


y = df[['exchange_type']].values
X = df.loc[:,df.columns != 'exchange_type'].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 2] = encoder.fit_transform(X[:, 2])

X[:, 15] = encoder.fit_transform(X[:, 15])
X[:, 16] = encoder.fit_transform(X[:, 16])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1007)
import numpy as np
from keras import utils

encoder2 = OneHotEncoder(sparse = False)
X_train=np.array(X_train).astype(np.float32)
y_train= encoder2.fit_transform(y_train)
y_train=np.array(y_train).astype(np.float32)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(50,  activation='relu'))
	model.add(Dense(4, activation='softmax'))
    
	# Compile model
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
	return model
opt = SGD(lr = 0.01)
ex_fit = KerasClassifier(build_fn = baseline_model, batch_size = 500, epochs=50)
ex_fit.fit(X_train,y_train)

