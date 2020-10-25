#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


data = pd.read_csv('/home/minhvu/Documents/Py4DS_Lab2/Py4DS_Lab2_Dataset/creditcard.csv')
for i in range (1,31):
    print(data.iloc[:,i].value_counts())
    print("*"*20)


# In[2]:


#Biểu đồ tần số cho 28V
V = data[[col for col in data.columns if 'V' in col]]

f, ax = plt.subplots(ncols = 2, nrows = 14, figsize=(15,2*len(V.columns)))


for i, c in zip(ax.flatten(), V.columns):
    sns.distplot(V[c], ax = i)

f.tight_layout()


# In[3]:


# heat map 
# the hien moi tuong quan cua cac features
# neu features co moi tuong quan lon => kiem tra moi tuong quan cua rieng 2 features nay de loc data
plt.figure(figsize=(30,30))
sns.heatmap(data.corr(), linewidths = 0.1, cmap = "YlGnBu", annot=True)


# In[4]:


#Kiểm tra min_max của scaler
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data)

data_norm = pd.DataFrame(np_scaled, columns = data.columns)

data_norm.head()


# In[5]:


#
data_melt = pd.melt(data_norm, id_vars=['Class'], value_vars=[col for col in data.columns if 'V' in col])

data_melt.iloc[0:1000000:100000,:]


# In[6]:


plt.figure(figsize=(15,10))

f, ax = plt.subplots(4, figsize = (15,10*4))

g1 = ['V'+str(i) for i in range(1,8)]
g2 = ['V'+str(i) for i in range(8,15)]
g3 = ['V'+str(i) for i in range(15,22)]
g4 = ['V'+str(i) for i in range(22,29)]

data_melt_1 = data_melt[data_melt['variable'].isin(g1)]
data_melt_2 = data_melt[data_melt['variable'].isin(g2)]
data_melt_3 = data_melt[data_melt['variable'].isin(g3)]
data_melt_4 = data_melt[data_melt['variable'].isin(g4)]

sns.violinplot(x="variable", y="value", hue="Class", data=data_melt_1, ax = ax[0], split=True)
sns.violinplot(x="variable", y="value", hue="Class", data=data_melt_2, ax = ax[1], split=True)
sns.violinplot(x="variable", y="value", hue="Class", data=data_melt_3, ax = ax[2], split=True)
sns.violinplot(x="variable", y="value", hue="Class", data=data_melt_4, ax = ax[3], split=True)

