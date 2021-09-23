# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:22:44 2021
#python tutorials for digital humanities: 
@author: sujitk
"""

import spacy
import json
import random

#read the data with json format
def load_data(file):
    with open (file, "r", encoding = "utf-8") as f:
        data = json.load()
    return (data)

#spacy pipe training
def train_spacy(train_data, iterations):
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    ner.add_label("new_label")
    nlp.add_pipe(ner,name = "new_label_ner")
    
    
    #other pipes running subsequently 
    optimizer = nlp.begin_training()
    for itn in range(iterations):
        print(f"starting iteration{str(itn)}")
        random.shuffle(train_data)
        losses = {}
        
        for text, annotations in train_data:
            nlp.update( [text],
                       [annotations],
                       drop = 0.2,
                       sgd = optimizer,
                       losses = losses
                       
                       )
            print(losses)
    return (nlp)
    
    
train_data = load_data("C:\users\sujitk....json")   
#print(train_data)


#limit number of training data
random.shuffle(train_data) 
train_data = train_data[0:100]

#train the model with n iterations
trained_model = train_spacy(train_data,5)

    
 #%%   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #https://www.youtube.com/channel/UC5vr5PwcXiKX_-6NTteAlXw