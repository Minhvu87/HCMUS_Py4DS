#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from scipy import stats
from scipy.stats import norm, skew #for some statistics

from sklearn import ensemble, tree, linear_model
#import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
#from plotly.colors import n_colors
#from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


data=pd.read_csv('/home/minhvu/Documents/Py4DS_Lab3/Py4DS_Lab3_Dataset/AB_NYC_2019.csv')


# # Exploring the data

# In[3]:


data.head()


# In[4]:


data.iloc[:,3:].describe()


# In[5]:


data.shape


# # Checking for null values

# In[6]:


data.isnull().sum()


# # Cleaning data

# # Remove missing data

# In[7]:


data.drop(['name','id','host_name','last_review'], axis=1, inplace=True)


# # fill null values in reviews_per_month by 0

# In[8]:


data['reviews_per_month'].fillna(0, inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


data.head()


# # EDA

# In[25]:


f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), linewidths = 0.1, cmap = "YlGnBu", annot=True)
plt.show()


# In[26]:


sns.pairplot(data)
plt.show()


# In[27]:


fig = plt.figure(figsize = (15,10))
ax = fig.gca()
data.hist(ax=ax)
plt.show()


# In[29]:


#room_type - price
result = data.groupby(["room_type"])['price'].aggregate(np.median).reset_index().sort_values('price')
sns.barplot(x='room_type', y="price", data=data, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[30]:


#neighbourhood_group - price
result = data.groupby(["neighbourhood_group"])['price'].aggregate(np.median).reset_index().sort_values('price')
sns.barplot(x='neighbourhood_group', y="price", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[32]:


labels = data.neighbourhood_group.value_counts().index
colors = ['green','yellow','orange','pink','red']
explode = [0,0,0,0,0]
sizes = data.neighbourhood_group.value_counts().values

plt.figure(0,figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Airbnb According to Neighbourhood Group',color = 'blue',fontsize = 15)
plt.show()


# In[34]:


plt.figure(figsize=(10,7))
sns.barplot(x = "neighbourhood_group", y = "price", hue = "room_type", data = data)
plt.xticks(rotation=45)
plt.show()


# In[36]:


plt.figure(figsize=(18,18))
sns.lmplot(x='minimum_nights',y='calculated_host_listings_count',hue="neighbourhood_group",data=data)
plt.xlabel('calculated_host_listings_count')
plt.ylabel('minimum_nights')
plt.title('calculated_host_listings_count vs minimum_nights')
plt.show()


# In[38]:


ax = sns.violinplot(x="neighbourhood_group", y="price",
                    data=data[data.price < 1000],
                    scale="width", palette="Set3")


# In[40]:


#neighbourhood_group - reviews_per_month
result = data.groupby(["neighbourhood_group"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')
sns.barplot(x='neighbourhood_group', y="reviews_per_month", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[41]:


#neighbourhood_group - minimum_nights
result = data.groupby(["neighbourhood_group"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')
sns.barplot(x='neighbourhood_group', y="minimum_nights", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[42]:


#neighbourhood_group - number_of_reviews
result = data.groupby(["neighbourhood_group"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')
sns.barplot(x='neighbourhood_group', y="number_of_reviews", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[43]:


ax = sns.violinplot(x="room_type", y="price",
                    data=data[data.price < 1000],
                    scale="width", palette="Set3")


# In[44]:


sns.kdeplot(data['price'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Price Kde Plot')
plt.show()


# In[45]:


#neighbourhood_group - reviews_per_month
result = data.groupby(["neighbourhood_group"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')
sns.barplot(x='neighbourhood_group', y="reviews_per_month", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[46]:


sns.lineplot(x='reviews_per_month',y='price',data=data)
plt.show()


# In[47]:


#neighbourhood_group - calculated_host_listings_count
result = data.groupby(["neighbourhood_group"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')
sns.barplot(x='neighbourhood_group', y="calculated_host_listings_count", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[48]:


sns.lineplot(x='calculated_host_listings_count',y='price',data=data)
plt.show()


# In[49]:


#neighbourhood_group - availability_365
result = data.groupby(["neighbourhood_group"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')
sns.barplot(x='neighbourhood_group', y="availability_365", data=data, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[50]:


sns.lineplot(x='availability_365',y='price',data=data)
plt.show()


# In[51]:


data.price.describe().T


# In[52]:


labels = data.room_type.value_counts().index
colors = ['orange','yellow','red']
explode = [0,0,0]
sizes = data.room_type.value_counts().values

plt.figure(0,figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Airbnb According to Room Type',color = 'blue',fontsize = 15)
plt.show()


# In[53]:


#room_type - minimum_nights
result = data.groupby(["room_type"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')
sns.barplot(x='room_type', y="minimum_nights", data=data, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[54]:


sns.lineplot(x='minimum_nights',y='price',data=data)
plt.show()


# In[55]:


#room_type - number_of_reviews
result = data.groupby(["room_type"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')
sns.barplot(x='room_type', y="number_of_reviews", data=data, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[56]:


sns.lineplot(x='number_of_reviews',y='price',data=data)
plt.show()


# In[57]:


#room_type - reviews_per_month
result = data.groupby(["room_type"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')
sns.barplot(x='room_type', y="reviews_per_month", data=data, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[58]:


#room_type - calculated_host_listings_count
result = data.groupby(["room_type"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')
sns.barplot(x='room_type', y="calculated_host_listings_count", data=data, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[59]:


#room_type - availability_365
result = data.groupby(["room_type"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')
sns.barplot(x='room_type', y="availability_365", data=data, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.show()


# In[60]:


#neighbourhood - price
plt.figure(figsize=(18,8))
#result = df.groupby(["neighbourhood"])['price'].aggregate(np.median).reset_index().sort_values('price')
sns.barplot(x=data.price[:25], y=data.neighbourhood[:25]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.xticks(rotation=90)
plt.show()


# In[61]:


#neighbourhood - minimum_nights
plt.figure(figsize=(18,8))
#result = df.groupby(["neighbourhood"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')
sns.barplot(x=data.minimum_nights[:25], y=data.neighbourhood[:25]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.xticks(rotation=90)
plt.show()


# In[62]:


#neighbourhood - number_of_reviews
plt.figure(figsize=(18,8))
#result = df.groupby(["neighbourhood"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')
sns.barplot(x=data.number_of_reviews[:25], y=data.neighbourhood[:25]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.xticks(rotation=90)
plt.show()


# In[63]:


#neighbourhood - reviews_per_month
plt.figure(figsize=(18,8))
#result = df.groupby(["neighbourhood"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')
sns.barplot(x=data.reviews_per_month[:25], y=data.neighbourhood[:25]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.xticks(rotation=90)
plt.show()


# In[64]:


#neighbourhood - calculated_host_listings_count
plt.figure(figsize=(18,8))
#result = df.groupby(["neighbourhood"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')
sns.barplot(x=data.calculated_host_listings_count[:25], y=data.neighbourhood[:25]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.xticks(rotation=90)
plt.show()


# In[65]:


#neighbourhood - availability_365
plt.figure(figsize=(18,8))
#result = df.groupby(["neighbourhood"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')
sns.barplot(x=data.availability_365[:25], y=data.neighbourhood[:25]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])
plt.xticks(rotation=90)
plt.show()


# In[67]:


data = data.sort_values(by=["price"], ascending=False)
data['rank']=tuple(zip(data.price))
data['rank']=data.groupby('price',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values
data.head()


# # Train test split

# In[11]:


feature_columns=['neighbourhood_group','room_type','price',
                 'minimum_nights','calculated_host_listings_count','availability_365']
all_data=data[feature_columns]


# In[18]:


all_data['room_type']=all_data['room_type'].factorize()[0]
all_data['neighbourhood_group']=all_data['neighbourhood_group'].factorize()[0]


# In[19]:


y = all_data['price']
x= all_data.drop(['price'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)


# # Modelling

# # Linear Regression

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


linreg = LinearRegression()
linreg.fit(x_train,y_train)
y_pred=(linreg.predict(x_test))

print('R-squared train score: {:.3f}'.format(linreg.score(x_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg.score(x_test, y_test)))


# # Ridge Regression

# In[22]:


from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(x_train, y_train)

y_pred=ridge.predict(x_test)

print('R-squared train score: {:.3f}'.format(ridge.score(x_train, y_train)))
print('R-squared test score: {:.3f}'.format(ridge.score(x_test, y_test)))


# # Lasso Regression

# In[23]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=10,max_iter = 10000)
lasso.fit(x_train, y_train)

print('R-squared score (training): {:.3f}'.format(lasso.score(x_train, y_train)))
print('R-squared score (test): {:.3f}'.format(lasso.score(x_test, y_test)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




