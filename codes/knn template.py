# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:06:29 2020

@author: sujitk
"""
import numpy as np
import pandas a pd

def get_data (limit = None):
    pring "Reading in and transforming data.."
    df = pd.read_csv('../train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:,1:]/255.0 #MNIST data
    y = data[:,0]
    if limit is not None:
        X, y = X[:limit], y[:limit]
    return X,y

def get_xor():
    X = np.zeros((200,2))
    X[:50] = np.random.random((50,2))/2 + 0.5
    X[50:100] = np.random.random((50,2))/2 
    X[100:150] = np.random.random((50,2))/2 + np.array([[0,0.5]])
    X[150] = np.random.random((50,2))/2 + np.array([[0.5,0]])
    y = np.array([0]*100+[1]*100)
    return X,y

    
