# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:50:45 2020

@author: sujitk
"""
import matplotlib.pyplot as plt
import csv
import matplotlib.ticker as ticker
time = []
download = []
upload = []

with open('test.csv','r') as csvfile:
    plots = csv.reader(csvfile,delimiter = ',')
    next(csvfile)
    for row in plots:
        if not (row):
            continue
        time.append(str(row[0]))
        download.append(float(row[1]))
        upload.append(float(row[2]))
        
print(time, "\n", download, "\n", upload)
     
plt.figure()
plt.plot(time,download,label = 'download',color='r')
plt.plot(time,upload,label = 'upload',color='b')

plt.xlabel('time')
plt.ylabel('speed in Mb/s')
plt.title('internet speed')
plt.legend()
#plt.savefig('test_graph.jpg',bbox_inches = 'tight')
     