# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:06:49 2020

@author: sujitk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:54:15 2019

@author: sujitk
"""

#importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#part 2 - Building ANN
#import keras libraries and packages
#import keras
from keras.models import Sequential  # initialize the NN 
from keras.layers import Dense # it is used to create the layers in the ANN

#initializing the ANN
classifier = Sequential()

#Adding the input and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#adding the second hidder layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)

#part 3 - making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#individual prediction
#new_prediction = classifier.predict(sc.transform(np.array([[0,0,600, 1,40,3,60000,2,1,1,50000]])))
#new_prediction = (new_prediction > 0.5)




#Part 4 - Evaluating, Improving and Tuning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential  # initialize the NN 
from keras.layers import Dense # it is used to create the layers in the ANN
def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
mean = accuracies.mean()
variances = accuracies.std()

#Improving the ANN
#Dropout regularization to reduce overfitting if needed
#initializing the ANN
classifier = Sequential()

#Adding the input and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

#adding the second hidder layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

#adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)



#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential  # initialize the NN 
from keras.layers import Dense # it is used to create the layers in the ANN
def build_classifier(optimizer):
        classifier = Sequential()
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

