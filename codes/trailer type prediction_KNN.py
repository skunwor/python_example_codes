# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:32:41 2020

@author: sujitk
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:26:03 2020

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
 """SELECT top (90000)
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
        ,spot_freight_chg
        ,freight_pm
        ,exchange_type
        ,dest_state
        ,origin_state
    FROM [DS01].[dbo].[x_rate_history] 
    where exclude_from_rate = 'N' and origin_yrweek is not null
    and exchange_type != 'O'
    union 
    
    SELECT top (90000)
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
        ,spot_freight_chg
        ,freight_pm
        ,exchange_type
        ,dest_state
        ,origin_state
    FROM [DS01].[dbo].[vw_x_rate_history_YTD] 
    where exclude_from_rate = 'N' and origin_yrweek is not null
    and exchange_type != 'O' and spot_freight_pm is not null
    """,conn)
    
    



from sklearn.model_selection import train_test_split
import numpy as np
from keras import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

df = df.dropna()

y = df[['exchange_type']].values
X = df.loc[:,df.columns != 'exchange_type'].values

encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 14] = encoder.fit_transform(X[:, 14])
X[:, 13] = encoder.fit_transform(X[:, 13])


scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1007)

X_train=np.array(X_train).astype(np.float32)
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_train=np.array(X_train).astype(np.float32)

#X_train[] = scaler.fit(X_train[])
y_train= encoder.fit_transform(y_train)
y_test= encoder.fit_transform(y_test)

#y_train = scaler.fit(y_train)

#y_train=np.array(y_train).astype(np.float32)



X_test=np.array(X_test).astype(np.float32)
X_test = pd.DataFrame(scaler.fit_transform(X_test))
X_test=np.array(X_test).astype(np.float32)





clf_knn2 = KNeighborsClassifier(n_neighbors = 6)
clf_knn2.fit(X_train,y_train)
confusion_matrix(y_test, clf_knn2.predict(X_test))
accuracy_score(y_test, clf_knn2.predict(X_test))








pd.Series(X_train[:,0]).isnull().any()
pd.Series(X_train[:,1]).isnull().any()
pd.Series(X_train[:,2]).isnull().any()
pd.Series(X_train[:,3]).isnull().any()
pd.Series(X_train[:,4]).isnull().any()
pd.Series(X_train[:,5]).isnull().any()
pd.Series(X_train[:,6]).isnull().any()
pd.Series(X_train[:,7]).isnull().any()
pd.Series(X_train[:,8]).isnull().any()

pd.Series(X_train[:,8]).isnull().any()

























