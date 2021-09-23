# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:26:50 2020

@author: sujitk
"""

class Animal (object):
    def __init__(self, age):
        self.age = age
        self.name = None
    def get_age(self):
        return self.age
    def get_name(self):
        return self.name
    def set_age (self,newage):
        self.age = newage
    def set_name (self, newname = ""):
        self.name = newname
    def __str__(self):
        return "animal:" +str(self.name)+":"+str(self.age)

class Cat(Animal):
    def speak(self):
        print("meow")
    def __str__(self):
        return "cat:" + str(self.name)+":"+str(self.age)
    
class Person(Animal):
    def __init__(self,name,age):
        Animal.__init__(self, age)
        self.set_name(name)
        self.friends = []
    def get_friends(self):
        return self.friends
    def add_friend(self, fname):
        if fname not in self.friends:
            self.friends.append(fname)
    def speak(self):
        print("hello")
    def age_diff(self,other):
        diff = self.age - other.age
        print(abs(diff), "year difference")
    def __str__(self):
        return "person:" + str(self.name)+":"+str(self.age)

