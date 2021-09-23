# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 09:41:20 2020

@author: sujitk
"""
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
#import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

data = pd.read_csv('C:/Users/sujitk/OneDrive - Tom McLeod Software, Inc/Desktop/case_study_data.csv')

#EDA
data.head()
data.shape
#lowercase words
data["l_text"] = data["text"].map(lambda x:x.lower())
data_ = data[["product_group","l_text"]]
#del(data)


# complains by product_group
fig,ax = plt.subplots(figsize=(18,6))
sns.countplot(x='product_group',data=data_)
data_.groupby('product_group').count().sort_values(['l_text'])

#text processing
#print(data_["l_text"][0].replace('x*x',''))
data_ = data_.replace(to_replace ='x*x', value = '', regex = True)
data_['l_text'] =data_['l_text'].str.replace(r'[^\w\s]',"")

stopword_list = nltk.corpus.stopwords.words('english')
data_['l_text'] =data_['l_text'].apply(lambda x: ' '.join([i for i in x.split() if i not in stopword_list]))


#data partitioning into training set and test (70:30) set using model_selection module
x_train, x_test,  y_train, y_test= train_test_split(data_['l_text'],data_['product_group'],
                        stratify = data_['product_group'],test_size =0.30)    


#target variable into label encodings
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


ct = TfidfVectorizer(analyzer = 'word', token_pattern=r'\w{2,}',max_features = 5000)
ct.fit(data_['l_text'])

#print(tfidf_vect.stop_words)
#print(tfidf_vect.vocabulary_)

train_tf_idf = ct.transform(x_train)
#print(train_tf_idf)
test_tf_idf = ct.transform(x_test)
#print(train_tf_idf)

#sort_orders = sorted(ct.vocabulary_.items())

log_reg = LogisticRegression().fit(train_tf_idf, y_train)
log_reg_predict = log_reg.predict(test_tf_idf)

logreg_accuracy = metrics.accuracy_score(log_reg_predict, y_test)
logreg_accuracy_report = metrics.classification_report(y_test,log_reg.predict(test_tf_idf), target_names=data_['product_group'].unique())
print(logreg_accuracy)
print(logreg_accuracy_report)


