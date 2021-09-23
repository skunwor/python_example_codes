# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:04:18 2020

@author: sujitk
"""

import pandas as pd
import numpy as np
import distance
df = pd.read_csv('location_dwell_scac.csv')
df= df.iloc[:,0]
df = np.array(df)

from fuzzywuzzy import fuzz

def get_ratio(row):
     name = row['Expedia']
    name1 = row['Booking.com']
    return fuzz.token_set_ratio(name, name1)

df[df.apply(get_ratio, axis=1) > 70].head(10)


lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in df] for w2 in df])
