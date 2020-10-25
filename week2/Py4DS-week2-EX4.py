#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
data = pd.read_csv('/home/minhvu/Documents/Py4DS_Lab2/Py4DS_Lab2_Dataset/HappinessReport2020.csv')
for i in range (1,20):
    print(data.iloc[:,i].value_counts())
    print("*"*20)


# In[26]:


print('Rows in Data: ', data.shape[0])
print('Columns in Data: ', data.shape[1])


# In[30]:


#Xem chỉ số hạnh phúc ladder core của top 10 nước có điểm cao nhất
data.head(10)


# In[3]:


# heat map 
# the hien moi tuong quan cua cac features
# neu features co moi tuong quan lon => kiem tra moi tuong quan cua rieng 2 features nay de loc data
plt.figure(figsize=(14,14))
sns.heatmap(data.corr(), linewidths = 0.1, cmap = "YlGnBu", annot=True)


# In[25]:


# plot label 
# kiem tra phan bo cua labels co deu hay khong?
# neu can bang=>co the su dung truc tiep 
# Biểu đồ thể hiện các nước được thống kê trong 1 vùng
P_satis = sns.countplot(x = "Regional indicator", data =data)
P_satis.set_xticklabels(P_satis.get_xticklabels(),rotation=90)


# In[20]:


#Bản đồ thể hiện được chỉ số hạnh phúc của các nước , màu càng sáng thì chứng tỏ nước đó càng hạnh phúc.
map_data = [go.Choropleth( 
           locations = data['Country name'],
           locationmode = 'country names',
           z = data["Ladder score"], 
           text = data['Country name'],
           colorbar = {'title':'Ladder Rank'})]

layout = dict(title = 'Least Satisfied Countries', 
             geo = dict(showframe = False, 
                       projection = dict(type = 'equirectangular')))

world_map = go.Figure(data=map_data, layout=layout)
iplot(world_map)


# In[33]:


# Bảng thể hiện các nước có chỉ số GGP bình quân cao nhất
GDP = data.sort_values(by='Logged GDP per capita',ascending=False)
GDP.head(10)


# In[28]:


#Bản đồ thể hiện chỉ số tự do các nước
map_data = [go.Choropleth( 
           locations = data['Country name'],
           locationmode = 'country names',
           z = data["Freedom to make life choices"], 
           text = data['Country name'],
           colorbar = {'title':'Ladder Rank'})]

layout = dict(title = 'Countries With Least Freedom', 
             geo = dict(showframe = False, 
                       projection = dict(type = 'equirectangular')))

world_map = go.Figure(data=map_data, layout=layout)
iplot(world_map)


# In[ ]:





# In[ ]:




