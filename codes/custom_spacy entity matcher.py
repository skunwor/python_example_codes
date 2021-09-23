# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 16:58:36 2021

@author: sujitk
"""
nlp = spacy.load("en_core_web_sm")

from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
pattern_ = [
    {"LEMMA" : "is"}, \
    {"POS":"DET","OP":"?"}, \
    {"POS":"ADJ"} \
    ]
matcher.add("id_caught",  [pattern_])
matcher(doc)



#%%
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp("ram is a great guy")

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern matching two tokens: "iPhone" and "X"
pattern = [{"LEMMA" : "be"},{"POS":"DET","OP":"?"},{"POS":"ADJ"}]

# Add the pattern to the matcher
matcher.add("new_pattern",  [pattern])


# Use the matcher on the doc


matches = matcher(nlp(df["review"][1]))



print("Matches:", [doc[start:end].text for match_id, start, end in matches])


#%%
nlp = spacy.load("en_core_web_sm")
doc = nlp("who is great guy")

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern matching two tokens: "iPhone" and "X"
pattern = [{"LEMMA":"be"}]
# Add the pattern to the matcher
matcher.add("new_pattern",  [pattern])
#matches = matcher(nlp(df["review"][1]))
matches = matcher(doc)

print("Matches:", [doc[start:end].text for match_id, start, end in matches])


for token in doc:
    print(token.dep_,token.text,token.)
    


#%%

from spacy.matcher import DependencyMatcher
nlp = spacy.load("en_core_web_sm")

pattern = [[
  # anchor token: be
  {
    "RIGHT_ID": "movie_is",
    "RIGHT_ATTRS": {"LEMMA": "be"}
  },
  # be -> subject
  {
    "LEFT_ID": "movie_is",
    "REL_OP": ">",
    "RIGHT_ID": "movie_name",
    "RIGHT_ATTRS": {"DEP": "nsubj"}
   },

  {
    "LEFT_ID": "movie_is",
    "REL_OP": ">",
    "RIGHT_ID": "movie_word",
    "RIGHT_ATTRS": {"POS": "NOUN"}
  }   
]]

matcher = DependencyMatcher(nlp.vocab,validate=True,)
matcher.add("movie_is", pattern)
text = "beyond long name, by new, is a movie"
doc = nlp(text)
matches = matcher(doc)  
for i in range(len(token_ids)):
    print(pattern[i]["RIGHT_ID"] + ":", doc[token_ids[i]].text)
 



for i in range(1,100):
    text = df["review"][i]
    matches = matcher(nlp(df["review"][1]))
    print("Matches:", [doc[start:end].text for match_id, start, end in matches])




#%%
#alternative approache --- look into thsi

nlp = spacy.load("en_core_web_sm")
matcher = DependencyMatcher(nlp.vocab)

pattern = [
    {
        "RIGHT_ID": "anchor_founded",
        "RIGHT_ATTRS": {"ORTH": "is"}
    },
    {
        "LEFT_ID": "anchor_founded",
        "REL_OP": ">",
        "RIGHT_ID": "founded_subject",
        "RIGHT_ATTRS": {"DEP": "nsubj"},
    },
    {
        "LEFT_ID": "anchor_founded",
        "REL_OP": ">",
        "RIGHT_ID": "founded_object",
        "RIGHT_ATTRS": {"ORTH": "movie"},
    }
]
text = "beyond long name, by new, is a movie"
doc = nlp(text)

matcher.add("FOUNDED", [pattern])
#doc = nlp("Lee, is a person's dog.")
matches = matcher(doc)

print(matches) # [(4851363122962674176, [6, 0, 10, 9])]
# Each token_id corresponds to one pattern dict
match_id, token_ids = matches[0]
for i in range(len(token_ids)):
    print(pattern[i]["RIGHT_ID"] + ":", doc[token_ids[i]].text)
    

for token in doc:
    print(token.dep_,token.text,token.pos_,token.tag_)
 

#%%
matcher = Matcher(nlp.vocab)
matched_sents = []  # Collect data of matched sentences to be visualized
pattern1 = [[ {"POS": "NOUN", "OP": "*"},{"DEP": {"IN": ["dobj","prep", "ccomp","attr","compound","nummod"]},
                                          "POS": {"IN":["NUM","NOUN","PROPN","VERB","ADP"]}, "OP": "*"},
           {"LEMMA": "be"}, {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}, {"LOWER": {"IN": ["film","movie"]}}],
           [{"LOWER": "film"},{"POS": {"IN",["VBG", "NOUN"]}, "OP": "*"},
            {"DEP": {"IN": ["det","dobj","prep", "ccomp","attr","compound","nummod"]},"POS": {"IN":["NUM","NOUN","PROPN","VERB","ADP"]}, "OP": "*"},
           {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}]
           ]
#matcher.add("ismovie", [pattern], on_match=collect_sents)  # add pattern
matcher.add("ismovie", pattern1, on_match=collect_sents)  # add pattern

doc = nlp("this film is a fancy one")
matches = matcher(doc)
displacy.render(matched_sents, style="ent", manual=True)


#%%
  # be -> object
  {
    "LEFT_ID": "movie_is",
    "REL_OP": ">",
    "RIGHT_ID": "movieobj",
    "RIGHT_ATTRS": {"DEP": "nobj"}
  }, {
    "LEFT_ID": "movie_is",
    "REL_OP": ">",
    "RIGHT_ID": "movie_adj",
    "RIGHT_ATTRS": {"DEP": "adj"}
  }, 




  ,
  # be -> object
  {
    "LEFT_ID": "movie_is",
    "REL_OP": ">",
    "RIGHT_ID": "movieobj",
    "RIGHT_ATTRS": {"DEP": "nobj"}
  }
,
  {
    "LEFT_ID": "movie_is",
    "REL_OP": ">",
    "RIGHT_ID": "movie_word",
    "RIGHT_ATTRS": {"ORTHO": "movie"}
  }    
   
  ,
[
  # anchor token: founded
  {
    "RIGHT_ID": "film",
    "RIGHT_ATTRS": {"ORTH": "film"}
  },
  # founded -> subject
  {
    "LEFT_ID": "film",
    "REL_OP": ">>",
    "RIGHT_ID": "subject",
    "RIGHT_ATTRS": {"DEP": "nsubj"}
  }
]]  
   
    
   
    
   
    
   
    
