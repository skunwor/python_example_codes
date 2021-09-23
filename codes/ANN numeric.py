# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:17:22 2020

@author: sujitk
"""

import pandas as pd

net_prop = pd.read_csv('~Documents\net_prop.csv')
net_prop1 = pd.read_csv("~/Documents/net_prop1.csv")
from keras.models import Sequential
from keras.layers import Dense
#first model
X_train = net_prop.loc[:,net_prop.columns != 'Autauga']
y_train = net_prop.loc[:, 'Autauga']
X_test = net_prop1.loc[:,net_prop1.columns != 'Autauga']
y_test = net_prop1.loc[:, 'Autauga']


model = Sequential([
    Dense(1000, activation='relu', input_shape=(74,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='linear'),
])
#opt = SGD(lr = 0.01, momentum = 0.9)
model.compile(optimizer='sgd',
              loss='mean_squared_error'
              )
hist = model.fit(X_train, y_train,batch_size = 22,
           epochs=100,validation_split = 0.3)
          
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#second model 
model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(74,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_2.compile(optimizer='adam',
              loss='mean_squared_error'
              )

hist_2 = model_2.fit(X_train, y_train,
          batch_size=32, epochs=50,validation_split = 0.3)


plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


plt.plot(hist_2.history['acc'])
plt.plot(hist_2.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


###### model with dropout
from keras.layers import Dropout
from keras import regularizers
model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.1), input_shape=(74,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01)),
])
model_3.compile(optimizer='adam',
              loss='mean_squared_error')

hist_3 = model_3.fit(X_train, y_train,
          batch_size=32, epochs=100,validation_split = 0.3)


plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()



plt.plot(hist_3.history['acc'])
plt.plot(hist_3.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()







