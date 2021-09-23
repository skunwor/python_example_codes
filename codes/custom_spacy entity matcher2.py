# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:15:36 2021

@author: sujitk
"""
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
matched_sents = []  # Collect data of matched sentences to be visualized

def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]  # Matched span
    sent = span.sent  # Sentence containing matched span
    # Append mock entity for match in displaCy style to matched_sents
    # get the match span by ofsetting the start and end of the span with the
    # start and end of the sentence in the doc
    match_ents = [{
        "start": span.start_char - sent.start_char,
        "end": span.end_char - sent.start_char,
        "label": "MOVIE",
    }]
    matched_sents.append({"text": sent.text, "ents": match_ents})

pattern = [[
           {"LEMMA": "be"}, {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}, {"LOWER": {"IN": ["movie"]}}]
           ]

pattern = [[ {"POS": "NOUN", "OP": "*"},{"DEP": {"IN": ["dobj","prep", "ccomp","attr","compound","nummod"]},
            "POS": {"IN":["NUM","NOUN","PROPN","VERB","ADP"]}, "OP": "*"},
           {"LEMMA": "be"}, {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}, {"LOWER": {"IN": ["film","movie"]}}],
           [ {"LOWER": "film"},{"POS": "VBG"},{"POS": "NOUN", "OP": "*"},
            {"DEP": {"IN": ["det","dobj","prep", "ccomp","attr","compound","nummod"]},"POS": {"IN":["NUM","NOUN","PROPN","VERB","ADP"]}, "OP": "*"},
           {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}
           ]]


from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
matched_sents = []  # Collect data of matched sentences to be visualized

#matcher.add("ismovie", [pattern], on_match=collect_sents)  # add pattern
matcher.add("ismovie", pattern, on_match=collect_sents)  # add pattern

doc = nlp("I'd say X-men, by someone, is fantastic movie . , the film is a great one. –  godfather two/three at is not so cool movie, right?")
matches = matcher(doc)
matched_sents
#displacy.render(matched_sents, style="ent", manual=True)
#,{"POS": "PROPN", "OP": "*"}


for i in range(1,100):
    text = df["review"][i]
    matches = matcher(nlp(df["review"][1]))
    print("Matches:", [doc[start:end].text for match_id, start, end in matches])


