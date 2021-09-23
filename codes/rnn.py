# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:02:10 2020

@author: sujitk
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


# training data import



# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScalar(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

for i in range(6,12):
    X_train.append(training_set_scaled[i-6:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)    

#reshaping
X_train = np.reshape(X_train,)
    
#building the RNN
