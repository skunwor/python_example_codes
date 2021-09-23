"""
Created on Sat Jul 25 09:41:20 2020

@author: sujitk
"""

print ("Logistic Regression > Accuracy: ", lr_accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test, final_lr_predict,target_names=data_['product_group'].unique()))
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, final_lr_predict)
fig, ax = plt.subplots(figsize=(16,16))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="BuPu",xticklabels=data_['product_group'].unique(),yticklabels=data_['product_group'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
#end of regularization


#SVM
svc_model = LinearSVC()
svc_params = {'C':[0.01,0.1, 1, 10, 100, 1000]}
grid_svc = GridSearchCV(estimator=svc_model,param_grid=svc_params,cv=5,n_jobs=-1)
grid_svc.fit(train_tfidf,y_train)

print(grid_svc.best_params_)
print(grid_svc.best_score_)

final_svc = LinearSVC(C=0.1)
final_svc.fit(train_tf_idf,y_train)


final_svc_predict = final_svc.predict(test_tfidf)
svc_accuracy = metrics.accuracy_score(final_svc_predict, y_test)
print ("SVC > Accuracy: ", svc_accuracy)


#XGBOOST

from xgboost import XGBClassifier
xgb_model = XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, eta=0.3, silent=1, subsample=0.8)
xgb_model.fit(train_tfidf, y_train)



xgb_predict = xgb_model.predict(test_tfidfl)
xgb_accuracy = metrics.accuracy_score(xgb_predict, valid_y)
print ("XGBoost > Accuracy: ", xgb_accuracy)


from sklearn.metrics import classification_report
print(classification_report(y_test, xgb_predict,target_names=data_['product_group'].unique()))





from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(valid_y, xgb_predict)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="BuPu",xticklabels=data_['product_group'].unique(),yticklabels=data_['product_group'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()





import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import nltk


import seaborn as sns
import matplotlib.pyplot as plt

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
nltk.download('wordnet')

stopword_list = nltk.corpus.stopwords.words('english')


df = pd.read_csv('C:/Users/sujitk/OneDrive - Tom McLeod Software, Inc/Desktop/case_study_data.csv')

df.head()
df.shape
df1 = df[["product_group","text"]]
fig,ax = plt.subplots(figsize=(18,6))
sns.countplot(x='product_group',data=df1)
df1.groupby('product_group').count().sort_values(['text'])



x_train, x_test,  y_train, y_test= train_test_split(df1['text'],df1['product_group'],
                        stratify = df1['product_group'],test_size =0.30)    
# Default it will split 25 by 75% means 25% test case and 75% training cases

##label encoding target variable


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Bidirectional, LSTM, Embedding, Dense

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

y_train = to_categorical(np.asarray(y_train))
y_test = to_categorical(np.asarray(y_test))
print('Shape of data tensor:', train_data.shape)
print('Shape of label tensor:', y_train.shape)
print('Shape of label tensor:', y_test.shape)

tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(x_train.values)
train_sequences = tokenizer.texts_to_sequences(x_train.values)
test_sequences = tokenizer.texts_to_sequences(x_test.values)


total_complaints = np.append(x_train.values,x_test.values)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
seq_max = max([len(c.split()) for c in total_complaints])
seq_max
model = Sequential()
train_data = pad_sequences(train_sequences, maxlen=seq_max,padding='post')
test_data = pad_sequences(test_sequences, maxlen=seq_max,padding='post')

wget http://nlp.stanford.edu/data/glove.6B.zip
GLOVE_DIR = '/mnt/data/temp/nlp/'


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

model.add(Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
model.add(layers.Bidirectional(LSTM(100, dropout_U = 0.2, dropout_W = 0.2)))
model.add(Dense(11,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


history = model.fit(train_data, y_train,
 batch_size=64,
 epochs=8,
 validation_data=(test_data, y_test))













ct = TfidfVectorizer(analyzer = 'word', token_pattern=r'\w{1,}',max_features = 7500)
ct.fit(df1['text'])

#print(tfidf_vect.stop_words)
#print(tfidf_vect.vocabulary_)

train_tfidf = ct.transform(x_train)
#print(xtrain_tfidf)
test_tfidf = ct.transform(x_test)
#sort_orders = sorted(ct.vocabulary_.items())
