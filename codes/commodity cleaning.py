# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:02:56 2020

@author: sujitk
"""



import nltk
import pandas as pd

data = pd.read_csv('C:/Users/sujitk/OneDrive - Tom McLeod Software, Inc/commodity list.csv')


lines = data['commodity'].str.cat(sep=', ')
# function to test if something is a noun
is_noun = lambda pos: pos[:2] == 'NN'
# do the nlp stuff
tokenized = nltk.word_tokenize(lines)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 

print(len(nouns))

