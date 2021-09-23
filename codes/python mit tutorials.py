# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:58:43 2020

@author: sujitk
"""

def printMove(fr, to):
    print ('move from ' + str(fr) + ' to '+ str(to))
    
def Towers(n,fr,to,spare):
    if n ==1:
        printMove(fr, to)
    else:
        Towers(n-1,fr,spare,to)
        Towers(1,fr,to,spare)
        Towers(n-1,spare,to, fr)
    
Towers(4,1,2,1)

def fib(x):
    if x == 0 or x ==1:
        return 1
    else:
        return fib(x-1) + fib(x-2)

fib(7)

def lyrics_to_frequencies(lyrics):
    myDict ={}
    for word in lyrics:
        if word in myDict:
            myDict[word] += 1
        else:
            myDict[word] = 1
    return myDict


def most_common_words(freqs):
    values = freqs.values()
    best = max(values)
    words = []
    for k in freqs:
        if freqs[k] == best:
            words.append(k)
    return(words, best)

    
class coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance (self, other):
        x_diff_sq = (self.x - other.x)**2
        y_diff_sq = (self.y - other.y)**2
        return (x_diff_sq + y_diff_sq)**0.5
    def __str__ (self):
        return "<"+str(self.x)+","+str(self.y)+">"


        
        
        