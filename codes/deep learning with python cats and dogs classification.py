# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:25:33 2020

@author: sujitk
"""

# pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
 

DATADIR = "C:\Petimages"
CATEGORIES = ["Dog","Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category) #path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap = "gray")
        plt.show()
        break
    break

#print(img_array.shape)     

img_size = 80
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap = 'gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category) #path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

print(len(training_data))

























