#!/usr/bin/env python
# coding: utf-8

# # 1 Importing Tequired Libraries

# In[1]:


#Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier#Import DecisionTreeClassifier
from sklearn.model_selection import train_test_split#Import train test split
from sklearn import metrics#Import scikit.learn metrics module for accuracy calculation
import eli5#Calculating and Displaying importance using the eli5 library
from eli5.sklearn import PermutationImportance


# # 2.Loading Data
# 

# In[2]:


col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
#Load dataset
df=pd.read_csv("Documents/Py4DS_Lab1_Dataset/diabetes.csv",header=0,names=col_names)

df.head()


# In[3]:


df.info()


# # 2.Feature Selection

# In[4]:


#split dataset in features and target variable
feature_cols=['pregnant','insulin','bmi','age','glucose','bp','pedigree']
X=df[feature_cols]#Features
y=df.label#Target variable


# # 3.Splitting Data

# In[5]:


#Split dataset into training set and test set
#70% training and 30% test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# # 4. Building Decision Tree Model

# In[6]:


#Create Decision Tree classifer object
clf=DecisionTreeClassifier()

#Train Decision Tree Classifer
clf=clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred=clf.predict(X_test)


# In[ ]:





# In[7]:


perm=PermutationImportance(clf,random_state=1).fit(X_test,y_test)
eli5.show_weights(perm,feature_names=X_test.columns.tolist())


# # 5.Evaluating Model

# In[8]:


#Model Accuracy, how often í the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# # 6.Visualizing Decision Trees

# In[17]:


from sklearn.tree import export_graphviz
#from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from IPython.display import Image
from sklearn import tree
#from sklearn.datasets import load_iris
#from os import system
import matplotlib.pyplot as plt
import pydotplus
#import sklearn
#import graphviz
#import os


# In[19]:


dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,
               filled=True,rounded=True,
               special_characters=True,feature_names=feature_cols,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#! dot -Tpng dot_data.dot -o diabetes.png
graph.write_png('diabetes.png')
Image(graph.create_png())
#import matplotlib.pyplot as plt
#import cv2
#%matplotlib inline
#img = cv2.imread('diabetes.png')
#plt.figure(figsize = (20, 20))
#plt.imshow(img)


# In[ ]:





# In[ ]:




