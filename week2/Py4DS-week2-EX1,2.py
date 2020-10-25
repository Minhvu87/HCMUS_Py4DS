#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

data = pd.read_csv('/home/minhvu/Documents/Py4DS_Lab1_Dataset/xAPI-Edu-Data.csv')
for i in range (1,17):
    print(data.iloc[:,i].value_counts())
    print("*"*20)


# In[2]:


sns.pairplot(data, hue = 'Class')


# In[3]:


# heat map 
# the hien moi tuong quan cua cac features
# neu features co moi tuong quan lon => kiem tra moi tuong quan cua rieng 2 features nay de loc data
plt.figure(figsize=(14,14))
sns.heatmap(data.corr(), linewidths = 0.1, cmap = "YlGnBu", annot=True)


# In[ ]:





# In[4]:


# plot label 
# kiem tra phan bo cua labels co deu hay khong?
# neu can bang=>co the su dung truc tiep duoc
P_satis = sns.countplot(x = "Class", data = data)


# In[5]:


# normalize label


# In[6]:


plt.figure(figsize=(20,14))
data.raisedhands.value_counts().sort_index().plot.bar()


# In[7]:


plt.figure(figsize=(10,10))
Raise_hand = sns.boxplot(x = "Class", y = "raisedhands", data=data)
plt.show()
# loai outliers


# In[8]:


Facegrid = sns.FacetGrid(data, hue = "Class")
Facegrid.map(sns.kdeplot, "raisedhands", shade = True)
Facegrid.set(xlim = (0, data.raisedhands.max()))


# In[9]:


# data.groupby
data.groupby(['ParentschoolSatisfaction'])['Class'].value_counts()


# In[10]:


pd.crosstab(data['Class'], data['ParentschoolSatisfaction'])


# In[11]:


# sns.countplot(ParentschoolSatis
# faction)
sns.countplot(x = "ParentschoolSatisfaction", data = data, hue = "Class")


# In[12]:


# pie chart
labels = data.ParentschoolSatisfaction.value_counts()
colors = ["blue", "green"]
explode = [0,0]
sizes = data.ParentschoolSatisfaction.value_counts().values

plt.pie(sizes, explode=explode, labels = labels, colors = colors)
plt.show


# In[ ]:




