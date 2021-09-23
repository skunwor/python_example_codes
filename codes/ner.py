# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:52:01 2021

@author: sujitk
"""

#%%
import pandas as pd
file = "C:/Users/sujitk/OneDrive - Tom McLeod Software, Inc/labeledTrainData.tsv"
df = pd.read_csv(file,sep = '\t', encoding = "utf-8")

df.head()

#%%
"""
from word2veckaggle import word2veckaggle
"""
 
#%%
import spacy
import en_core_web_sm

nlp = spacy.load("en_core_web_sm") # small english model 10MB -- large 790MB 1% diff in fscore, precision and recall

nlp1 = en_core_web_sm.load()



review_entities = []
for i in range(0,100):
    review_entities.append([(X.text, X.label_) for X in nlp(str(df["review"][i])).ents])

   



#%%
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

movie_names = pd.read_csv('C:/Users/sujitk/OneDrive - Tom McLeod Software, Inc/Desktop/Convolutional_Neural_Networks/IMDb movies.csv')
movie_names_eng = movie_names[movie_names["language"].str.lower() == "english"]
movie_names_titles = movie_names_eng["original_title"]

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.extend(['or','&'])

movie_names_titles_stop = movie_names_titles.apply(lambda x: ' '.join([i for i in x.split() if i not in stopword_list]))

cv = CountVectorizer(ngram_range=(3,7))
ngrams = cv.fit_transform(movie_names_titles_stop)
def getList(dict):
    return [*dict]
vocab = cv.vocabulary_
movie_names_titles_upd = pd.Series(getList(vocab))

final_movie_ngrams = movie_names_titles.append(movie_names_titles_upd)
del(movie_names_titles_stop,movie_names_titles_upd)

final_movie_ngrams = final_movie_ngrams.drop_duplicates()
movies_ngram = []
for movies in final_movie_ngrams:
    movies_ngram.append(movies)

#%%

import spacy
from spacy.lang.en import English
#from spacy.pipeline import EntityRuler
import json

def create_training_data(data, type):
    data = data
    patterns = []
    for item in data:
        pattern = {
                    "label": type,
                    "pattern": item
                    }
        patterns.append(pattern)
    return (patterns)

def generate_rules(patterns):
    nlp = English()
    #nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    #nlp.add_pipe(ruler)
    nlp.to_disk("hp_ner")

def test_model(model, text):
    doc = model(text)
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return (results)

def save_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

        
#patterns = create_training_data(movies_ngram,"MOVIE")
#patterns = create_training_data(movie_names_titles,"MOVIE")
#generate_rules(patterns)

nlp = spacy.load("hp_ner")


pre_tag_data= {}
for i in range(1,100):
    text = df["review"][i]
    results = test_model(nlp,text)
    mov = []
    for result in results:
        mov.append(result)
    pre_tag_data[i]= mov

save_data("pre_tag_data.json", pre_tag_data)







     


 
#%%
"""
clean_train_reviews=[]

for i in range(1,5):
    clean_train_reviews.append(" ".join(word2veckaggle.review_to_wordlist(df["review"][i],True)))
    
    
   
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')

print([(X.text, X.label_) for X in doc.ents])
"""
"""print ("Cleaning and parsing the training set movie reviews...\n")
for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
"""

"""
print([(X, X.ent_iob_, X.ent_type_) for X in doc])
a =dict([(str(x), x.label_) for x in nlp(str(df["review"][1])).ents])
print([(x, x.ent_iob_, x.ent_type_) for x in nlp(str(df["review"]))])

a = [(x, x.ent_iob_, x.ent_type_) for x in nlp(str(df["review"]))]


reviews = []
for i in range(0,len(df["review"])):
    reviews.append(" ".join(word2veckaggle.review_to_wordlist(df["review"][i],True)))

a = [(x, x.ent_iob_, x.ent_type_) for x in nlp(str(reviews[1:1000]))]


#displacy.render(nlp(str(df["review"][1])), jupyter=True, style='ent')

html_obj  = displacy.render(nlp(str(df["review"][1])), style='dep', options = {'distance': 120})

#displacy.serve(doc)


# Write html object to a file (adjust file path; Windows path is used here)
with open('C:/users/sujitk/ner_test.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = r'C:/users/sujitk/ner_test.htm'
webbrowser.open(url, new=2)

"""











