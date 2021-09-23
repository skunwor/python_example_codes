# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:57:34 2020

@author: sujitk
"""

# this part is to regularize the model
from sklearn.model_selection import GridSearchCV
import numpy as np
log_reg1 = LogisticRegression()
x = range(1,10)
log_reg_params = {'C':[c for c in x]}
grid_search = GridSearchCV(estimator=log_reg1,param_grid=log_reg_params,cv=5,n_jobs = -1)
grid_search.fit(train_tf_idf,y_train)
print(grid_search.best_params_)
# the value of c was used below to regularize the model
log_reg1 = LogisticRegression(C=3)
log_reg1.fit(train_tf_idf,y_train)

log_reg_predict_1 = log_reg1.predict(test_tf_idf)
logreg_accuracy_1 = metrics.accuracy_score(log_reg_predict_1, y_test)
logreg_accuracy_report_1 = metrics.classification_report(y_test,log_reg1.predict(test_tf_idf), target_names=data_['product_group'].unique())
print(logreg_accuracy_1)
print(logreg_accuracy_report_1)