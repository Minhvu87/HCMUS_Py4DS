#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[3]:


path='Documents/Py4DS_Lab1_Dataset/spam.csv'
#Read in the data into a pandas dataframe
dataset_pd=pd.read_csv(path)
#Read in the data into a numpy array
dataset_np=np.genfromtxt(path,delimiter=',')


# In[4]:


X=dataset_np[:,0:len(dataset_np[0])-1]
y=dataset_np[:,len(dataset_np[0])-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)


# # Compute the metrics

# In[5]:


#clf=DecisionTreeClassifier(criterion='entropy')
clf=DecisionTreeClassifier(criterion='entropy')


# In[6]:


#fit Decision Tree Classifier
clf=clf.fit(X_train,y_train)
#Predict testset
y_pred=clf.predict(X_test)
#Evaluate performance of the model
print("CART (Tree Prediction) Accuracy: {}".format(sum(y_pred==y_test)/len(y_pred)))
print("CART (Tree Prediction) Accuracy by calling metrics: ",metrics.accuracy_score(y_test,y_pred))


# # 5-fold cross validation

# In[7]:


#Evaluate ascore by cross-validation
scores=cross_val_score(clf,X,y,cv=5)
print("scores={} \n final score={} \n".format(scores,scores.mean()))
print("\n")


# In[ ]:





# In[8]:


clf=SVC()

#Fit SVM Classifier
clf.fit(X_train,y_train)
#Predict testset
y_pred=clf.predict(X_test)
#Evaluate performance of the model
print("SVM Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")
#Evaluate a score by cross-validation
scores=cross_val_score(clf,X,y,cv=5)
print("scores={}\n final score={}\n".format(scores,scores.mean()))
print("\n")


# # Random Forest

# In[9]:


#Fit Random Forest Classifier
rdf=RandomForestClassifier()
rdf.fit(X_train,y_train)
#Predict testset
y_pred=rdf.predict(X_test)
#Evaluate performance of the model
print("RDF: ",metrics.accuracy_score(y_test,y_pred))
print("\n")
#Evalute a score by cross validation
scores=cross_val_score(rdf,X,y,cv=5)
print("scores={}\n final score={}\n".format(scores,scores.mean()))
print("\n")


# # Logistic Regression

# In[16]:


#Fit Logistic Regression Classifier
#lr=LogisticRegression()
lr = LogisticRegression( solver='liblinear', multi_class='ovr',class_weight='balanced',)
lr.fit(X_train,y_train)
#Predict testset
y_pred=lr.predict(X_test)
#Evaluate performance of the model
print("LR: ",metrics.accuracy_score(y_test,y_pred))
#Evalute a score by cross validation
scores=cross_val_score(lr,X,y,cv=5)

print("scores={} \n final score=() \n".format(scores,scores.mean()))
print("\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




