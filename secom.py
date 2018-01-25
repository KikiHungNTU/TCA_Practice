# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:24:11 2018

@author: Ouch
"""

import pandas as pd

path = 'C:/Users/chich/Desktop/secomData/'

dataFile = path + 'secom.txt'
labelFile = path + 'secom_labels.txt'

all_data = pd.read_csv(dataFile,delim_whitespace = True,header = None, dtype = float,na_values = 'NaN')
#labels with time stamp
labels = pd.read_csv(labelFile,delim_whitespace = True,names = ['label', 'date'],na_values = 'NaN',keep_date_col = True)

#print(labels['date'][0])
#print(labels['date'][0].split(' ')[0])

data_date_time = []
for i in range(len(labels)):
    d = labels['date'][i].split(' ')[0]
    t = labels['date'][i].split(' ')[1]
    data_date_time.append([0,0])
    data_date_time[i][0] = d
    data_date_time[i][1] = t
