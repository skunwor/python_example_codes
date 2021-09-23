# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:03:08 2021

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
 """SELECT distinct [id],scac,[origin_yrweek],[exclude_from_spot],[exclude_from_rate],
    [exclude_rate_reason],[exclude_spot_reason],[freight_chg],[freight_pm],
    [spot_freight_chg],[spot_freight_pm],[pay_distance],[origin_zip_three],[origin_market_id],
    [dest_zip_three],[dest_market_id],origin_actual_av, [exchange_type],commodity
    FROM [DS01].[dbo].[vw_x_rate_history] 
    where month(origin_actual_av) between 10 and 12
    """,conn)
    
    



from sklearn.model_selection import train_test_split
import numpy as np
from keras import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd
import glob
path = r"C:\Users\sujitk\Downloads\history_2020_12_19\*.csv"
for fname in glob.glob(path):
   df=pd.read_csv(fname)
   my_list=list(df.columns)
   print(len(my_list),my_list)

# y = df[['exchange_type']].values
# X = df.loc[:,df.columns != 'exchange_type'].values











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

























