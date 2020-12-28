#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[2]:


path1='/home/minhvu/Documents/Py4DS_Lab6/Py4DS_Lab6_Dataset/Santander_train.csv'
path2='/home/minhvu/Documents/Py4DS_Lab6/Py4DS_Lab6_Dataset/Santander_test.csv'
X_train=pd.read_csv(path1)
X_test=pd.read_csv(path2)


# In[3]:


target = X_train['TARGET']


# In[4]:


target.value_counts()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(dpi=100)
sns.set_style('darkgrid')
sns.countplot(target, data=target)
plt.xlabel('target')
plt.ylabel('Count')
plt.xticks([0,1],['Satisfied','Not Satisfied'])
plt.show()


# In[6]:


# drop TARGET label from X_train

X_train.drop(labels=['TARGET'], axis=1, inplace = True)


# In[7]:


X_train.shape, X_test.shape


# In[8]:


#Checking Missing Values
X_train.isnull().sum().any()


# # 1.Constant Features

# In[9]:


#using sklearn variancethreshold to find constant features

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
sel.fit(X_train)  # fit finds the features with zero variance


# In[10]:


# get_support is a boolean vector that indicates which features are retained
# if we sum over get_support, we get the number of features that are not constant
sum(sel.get_support())


# In[11]:


constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[sel.get_support()]]

print(len(constant_columns))


# In[12]:


for column in constant_columns:
    print(column)


# In[13]:


constant_columns


# In[14]:


X_train = X_train.drop(constant_columns,axis=1)


# In[15]:


X_train.shape


# In[16]:


# 2: Feature Selection- With Correlation

import matplotlib.pyplot as plt
import seaborn as sns

#Using Pearson Correlation
corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(12,12)
sns.heatmap(corrmat,cmap="CMRmap_r")


# In[17]:


# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[18]:


corr_features = correlation(X_train, 0.98)
len(set(corr_features))


# In[19]:


corr_features


# In[20]:


X_train = X_train.drop(corr_features,axis=1)


# In[21]:


X_train.shape


# In[22]:


# sns.heatmap(X_train.corr());

corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(12,12)
sns.heatmap(corrmat,cmap="CMRmap_r")


# # Feature Scaling

# In[23]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)


# In[24]:


X_train.head()


# In[25]:


print((X_train.shape,target.shape))


# In[26]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(X_train,target,test_size=0.80,random_state=0,stratify=target)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # SMOTE technique

# In[28]:


from imblearn.over_sampling import SMOTE
# print(imblearn.__version__)
oversample = SMOTE()
X, y = oversample.fit_resample(X_train, target)


# In[29]:


print(X.shape,y.shape)


# In[30]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(X,y,test_size=0.80,random_state=0,stratify=y)


# In[31]:


from sklearn.linear_model import LogisticRegression
lrm=LogisticRegression(C=0.1,penalty='l2',n_jobs=-1)
lrm.fit(x_train,y_train)


# In[32]:


y_pred=lrm.predict(x_test)
y_pred


# In[33]:


from sklearn import metrics
print("Train Set Accuracy is ==> ",metrics.accuracy_score(y_train,lrm.predict(x_train)))


# In[34]:


from sklearn.metrics import confusion_matrix
confusion_matrix (y_train,lrm.predict(x_train))


# # PCA

# In[35]:


from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
pca.explained_variance_ratio_


# In[36]:


#  4:  PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
pca_resampled_test = PCA().fit(X_train)

# plot the Cumulative Summation of the Explained Variance for the different number of components
plt.figure()
plt.plot(np.cumsum(pca_resampled_test.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()


# In[37]:


# instantiate PCA
pca = PCA(n_components=150)

# fit PCA
principalComponents = pca.fit_transform(X_train)


# In[38]:


train_pc = pd.DataFrame(data = principalComponents)
train_target = pd.Series(target, name='TARGET')

train_pc_df = pd.concat([train_pc, train_target], axis=1)
train_pc_df.head(5)


# In[39]:


sns.heatmap(train_pc.corr())


# In[40]:


# we calculate the variance explained by priciple component
print('Variance of each component:', pca.explained_variance_ratio_)
print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))


# Logistic regression after applying PCA

# In[41]:


train_pc_df.shape , train_target.shape


# In[42]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(train_pc_df,train_target,test_size=0.80,random_state=0,stratify=train_target)


# In[43]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print('Logistic Regression accuracy score with the first 150 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[44]:


from sklearn.metrics import confusion_matrix
confusion_matrix (y_test, y_pred)


# In[ ]:




