# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 23:31:05 2018

@author: Ouch
"""

import pandas as pd
import numpy as np

path = 'C:/Users/chich/Desktop/Graduate/TCA/TestingData/'
dataFile = pd.read_csv(path + 'DATA.csv', nrows = 5000)
data = pd.read_csv(path + 'DATA.csv', nrows = 5000).as_matrix()
data = data[:, 303:603]
data = pd.DataFrame(data)

wafer_id = dataFile['Wafer.ID']
wafer_yield = dataFile['Yield']
wafer = pd.concat( [wafer_id, wafer_yield ], axis = 1 )
wafer = pd.concat( [wafer, data], axis = 1)

wafer_yield = pd.DataFrame(wafer_yield)
#Normalized the yield
yield_mean = wafer_yield.mean()
yield_std = wafer_yield.std()
wafer_yield.head()

#Scale
yield_max = wafer_yield.max()
yield_min = wafer_yield.min()

wafer_yield_scaled = (wafer_yield - yield_min) / (yield_max - yield_min)
wafer_yield_scaled = pd.DataFrame(wafer_yield_scaled)
count = wafer_yield_scaled.groupby('Yield').size()
#wafer_yield.plot.hist(figsize = (15,15), fontsize = 14,xticks = range(60,90,2),bins=100 ,grid = True,edgecolor="k", legend = False )

#K means
from sklearn.cluster import KMeans

kmeans_fit = KMeans(n_clusters = 2).fit(wafer_yield)
# 印出分群結果
cluster_labels = kmeans_fit.labels_
print("分群結果：")
print(cluster_labels)
print("---")

cluster_labels = pd.DataFrame(cluster_labels, columns = ['labels'], dtype = int)
wafer_yield_KMeans = pd.concat([wafer_yield, cluster_labels], axis = 1)

wafer_yield_KMeans_sorted = wafer_yield_KMeans.sort_values('Yield')

clf = (wafer_yield_KMeans['Yield'][1513] + wafer_yield_KMeans['Yield'][2971] )/2

wafer_KMeans = pd.concat([wafer_yield_KMeans, data], axis = 1)
#0=Good; 1=Bad

import matplotlib.pyplot as plt
for i in range(1000):
    if wafer_yield_KMeans['labels'][i] == 0:
        plt.scatter( i,wafer_yield_KMeans['Yield'][i]  ,c = 'red',s = 5)
    else:
        plt.scatter( i, wafer_yield_KMeans['Yield'][i] ,c = 'blue',s = 5)
plt.xlabel('Yield')
plt.ylabel('data')
plt.show()
        
#len(wafer_yield_KMeans)
#wafer_yield_KMeans['Yield'][0]   


#    if clusters[i] == 1:
#        plt.scatter(visual_encoded__[i,56],visual_encoded__[i,82],c = 'green',s = 6)
#    else:
#        plt.scatter(visual_encoded__[i,56],visual_encoded__[i,82],c = 'blue',s = 6)
#        
#plt.show()
    
    
machines = np.unique(data)
machines = np.array(machines)

machines_performance = np.zeros( (2,233) )
machines_performance = pd.DataFrame(machines_performance, columns = machines )

#wafer_KMeans.shape

#machines_performance[wafer_KMeans[299][4570] ][1]
#wafer_KMeans['labels'][4]
import time

start_time = time.time()

for i in range(5000):
    for j in range(0, 300):
        label = wafer_KMeans['labels'][i]
        machine = wafer_KMeans[j][i]
        if label == 0:
            #GOOD            
            machines_performance[machine][0] += 1
        else:
            #Bad
            machines_performance[machine][1] += 1

end_time = time.time()

t = end_time - start_time

print('---Spend ' + str(t) + ' Seconds---')

#good_machine = []
#bad_machine = []
#
#for i in range(5000):
#    for j in range(0, 300):
#        label = wafer_KMeans['labels'][i]
#        machine = wafer_KMeans[j][i]
#        if label == 0:
#            #GOOD
#            good_machine.append([machine])
#        else:
#            bad_machine.append([machine])

numberOfCounts = wafer_KMeans.groupby('labels').size()
numberOfBad = numberOfCounts[1] - 1

len(machines_performance.T)
bad_machine = []
good_machine = []
for i in range( len(machines_performance.T) ):
    #BAD machine
    if machines_performance.values[1][i] > numberOfBad:
        bad_machine.append([machines[i]])
    else:
        good_machine.append([machines[i]])

np.shape(bad_machine)

machines_yield_BYlabel = np.zeros((1,233))
#Compute all yield
for i in range( len(machines_performance.T) ):
    machines_yield_BYlabel[0][i] = machines_performance.values[0][i] / (machines_performance.values[0][i] + machines_performance.values[1][i])

machines_dict = {}
for i in range(len(machines)):
    machines_dict[ machines[i] ] = i
    
machine_yield_BYwafer = []
for i in range(233):
    machine_yield_BYwafer.append([])
#machine_yield_BYwafer[3].append(69)

for i in range(5000):
    for j in range(0, 300):
        yield_ = wafer['Yield'][i]
        machine_ = wafer[j][i]
        machine_idx = machines_dict[ machine_ ]
        machine_yield_BYwafer[machine_idx].append( yield_)

test = np.array(machine_yield_BYwafer[0])
test_df = pd.DataFrame(test.T)
test_df


for i in range(len(machine_yield_BYwafer)):
    temp = np.array(machine_yield_BYwafer[i])
    temp_df = pd.DataFrame(temp.T)
    test_df = pd.concat([test_df, temp_df], axis = 1)

test_df = test_df.iloc[:,1:234]
test_df.columns = machines_dict.keys()

bad_machines = np.array(bad_machine)
good_machines = np.array(good_machine)
bad_machine_dict = {}
good_machine_dict = {}
#for i in range(len(bad_machines)):
#    bad_machine_dict[ bad_machines[i][0] ] = i
for i in range(len(bad_machines)):
    bad_machine_dict[i] = str(bad_machines[i][0])


for i in range(len(good_machines)):
    good_machine_dict[i] = str(good_machines[i][0])

bad_machine_dict[1]
bad_df = pd.DataFrame(test_df['ToolA1'])
good_df = pd.DataFrame(test_df['ToolA10'] )
good_df.columns = ['Good']

#split GOOD and BAD machines
for j in range(1,len(bad_machine_dict)):
     bad_df = pd.concat([bad_df, test_df[ bad_machine_dict[j] ] ] , axis = 1)
     #good_df = good_df.drop([ bad_machine_dict[j] ] , axis = 1)

for i in range(1,len(good_machine_dict)):
    good_df = pd.concat([good_df, test_df[ good_machine_dict[i] ] ] , axis = 0)

good_df.dropna()
del good_df[0]
bad_df['ToolA1'].dropna()
good_df['Good']

from scipy.stats import ttest_ind
t, p = ttest_ind(bad_df['ToolA1'].dropna(), good_df['Good'].dropna(), equal_var=False)
t
p

t_p = np.zeros((2, 108))
for i in range(len(bad_machine_dict)):
    bad = bad_machine_dict[i]
    t, p = ttest_ind(bad_df[ bad ].dropna(), good_df['Good'].dropna(), equal_var=False)
    t_p[0][i] = t
    t_p[1][i] = p

bad_tp_df = pd.DataFrame(t_p, columns = bad_machine_dict.keys() )


df_draw = pd.concat([bad_df['ToolA4'].dropna(),good_df['Good'].dropna()], axis = 1)
df_draw.dropna().plot.box(patch_artist=False,meanline=False,showmeans=True)

    