#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy.stats import skew

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('/home/minhvu/Documents/Py4DS_Lab3/Py4DS_Lab3_Dataset/FIFA2018Statistics.csv')


# In[2]:


data


# In[3]:


y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)


# In[4]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# In[5]:


numerical_features   = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns


# In[6]:


numerical_features


# In[7]:


categorical_features


# In[ ]:





# In[8]:


data.describe()


# In[9]:


data.hist(figsize=(30,30))
plt.plot()


# In[10]:


skew_values = skew(data[numerical_features], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)


# In[11]:


# Missing values
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


# # EDA

# In[12]:


# encode target variable 'Man of the match' into binary format
data['Man of the Match'] = data['Man of the Match'].map({'Yes': 1, 'No': 0})


# In[13]:


sns.countplot(x = 'Man of the Match', data = data)


# In[14]:


plt.figure(figsize=(30,10))
sns.heatmap(data[numerical_features].corr(), square=True, annot=True,robust=True, yticklabels=1)


# In[15]:


var = ['Man of the Match','Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
corr = data.corr()
corr = corr.filter(items = ['Man of the Match'])
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)


# In[16]:


var = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
plt.figure(figsize=(15,10))
sns.heatmap((data[var].corr()), annot=True)


# In[17]:


var1 = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed']
var1.append('Man of the Match')
sns.pairplot(data[var1], hue = 'Man of the Match', palette="husl")
plt.show()


# # Outliers detection and removal

# In[18]:


dummy_data = data[var1]
plt.figure(figsize=(20,10))
sns.boxplot(data = dummy_data)
plt.show()


# # Missing values treatment

# In[19]:


data.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)


# # Categorical features encoding

# In[20]:


def uniqueCategories(x):
    columns = list(x.columns).copy()
    for col in columns:
        print('Feature {} has {} unique values: {}'.format(col, len(x[col].unique()), x[col].unique()))
        print('\n')
uniqueCategories(data[categorical_features].drop('Date', axis = 1))


# In[21]:


data.drop('Date', axis = 1, inplace=True)


# In[22]:


data.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)


# 

# In[23]:


print(data.shape)
data.head()


# In[24]:


cleaned_data  = pd.get_dummies(data)


# In[25]:


print(cleaned_data.shape)
cleaned_data.head()


# # Train test split

# In[41]:


y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=10000)


# In[ ]:





# # Linear Regression

# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred=(linreg.predict(X_test))

print('R-squared train score: {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg.score(X_test, y_test)))

