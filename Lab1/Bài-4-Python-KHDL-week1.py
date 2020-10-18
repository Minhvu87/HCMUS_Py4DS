#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df=pd.read_csv('Documents/Py4DS_Lab1_Dataset/xAPI-Edu-Data.csv')
df.head()


# In[5]:


y=df['gender']
X=df.drop('gender',axis=1)
y.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column]=labelencoder.fit_transform(df[column])


# In[7]:


df.dtypes


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
df_div=pd.melt(df,"gender",var_name="School")
fig,ax=plt.subplots(figsize=(10,5))
p=sns.violinplot(ax=ax,x="School",y="value",hue="gender",split=True,data=df_div,inner="quartitle")
df_no_class=df.drop(["gender"],axis=1)
p.set_xticklabels(rotation=90,labels=list(df_no_class.columns))


# In[11]:


plt.figure()
pd.Series(df['gender']).value_counts().sort_index().plot(kind='bar')
plt.ylabel("Counts")
plt.xlabel("gender")
plt.title('Number of male/female (0=female, 1=male)')


# In[12]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=1,cmap="YlGnBu",annot=True)
plt.yticks(rotation=0)


# # Model,predict and estimate the result:

# In[14]:


#Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier#Import DecisionTreeClassifier
from sklearn.model_selection import train_test_split#Import train test split
from sklearn import metrics#Import scikit.learn metrics module for accuracy calculation
import eli5#Calculating and Displaying importance using the eli5 library
from eli5.sklearn import PermutationImportance
#feature_cols=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size',
#            'gill-color','stalk-shape','stalk-root''stalk-surface-above-ring','stalk-surface-below-ring',
 #             'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
  #            'spore-print-color','population''habitat')])]
X=df.drop(['gender'],axis=1)
Y=df['gender']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)


# In[15]:


#Create Decision Tree classifer object
clf=DecisionTreeClassifier()

#Train Decision Tree Classifer
clf=clf.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred=clf.predict(X_test)


# In[16]:


perm=PermutationImportance(clf,random_state=1).fit(X_test,Y_test)
eli5.show_weights(perm,feature_names=X_test.columns.tolist())


# # Compute the metrics

# In[17]:


#Create Decision Tree classifer object
clf=DecisionTreeClassifier()

#Train Decision Tree Classifer
clf=clf.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred=clf.predict(X_test)


# # Evaluating Model

# In[18]:


#Model Accuracy, how often í the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))


# # 5-fold cross validation

# In[19]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
clf=SVC()
#Evaluate ascore by cross-validation
scores=cross_val_score(clf,X,Y,cv=5)
print("scores={} \n final score={} \n".format(scores,scores.mean()))
print("\n")

#Fit SVM Classifier
clf.fit(X_train,Y_train)
#Predict testset
Y_pred=clf.predict(X_test)
#Evaluate performance of the model
print("SVM Accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
print("\n")
#Evaluate a score by cross-validation
scores=cross_val_score(clf,X,Y,cv=5)
print("scores={}\n final score={}\n".format(scores,scores.mean()))
print("\n")


# # Random Forest

# In[20]:


from sklearn.ensemble import RandomForestClassifier
#Fit Random Forest Classifier
rdf=RandomForestClassifier()
rdf.fit(X_train,Y_train)
#Predict testset
Y_pred=rdf.predict(X_test)
#Evaluate performance of the model
print("RDF accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
print("\n")
#Evalute a score by cross validation
scores=cross_val_score(rdf,X,Y,cv=5)
print("scores={}\n final score={}\n".format(scores,scores.mean()))
print("\n")


# # Logistic Regression

# In[21]:


from sklearn.linear_model import LogisticRegression
#Fit Logistic Regression Classifier
#lr=LogisticRegression()
lr = LogisticRegression( solver='liblinear', multi_class='ovr',class_weight='balanced',)
lr.fit(X_train,Y_train)
#Predict testset
Y_pred=lr.predict(X_test)
#Evaluate performance of the model
print("LR: ",metrics.accuracy_score(Y_test,Y_pred))
#Evalute a score by cross validation
scores=cross_val_score(lr,X,Y,cv=5)

print("scores={} \n final score=() \n".format(scores,scores.mean()))
print("\n")


# # Visualizing Decision Trees

# In[23]:


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


# In[30]:


#Create Decision Tree classifer object
clf=DecisionTreeClassifier()
#Train Decision Tree Classifer
clf=clf.fit(X_train,Y_train)
#Predict the response for test dataset
Y_pred=clf.predict(X_test)
feature_cols=['NationalITy','PlaceofBirth','StageID','GradeID',
              'SectionID','Topic','Semester','Relation','raisedhands',
              'VisITedResources','AnnouncementsView','Discussion',
              'ParentAnsweringSurvey','ParentschoolSatisfaction',
              'StudentAbsenceDays','Class']
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,
               filled=True,rounded=True,
               special_characters=True,feature_names=feature_cols,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#! dot -Tpng dot_data.dot -o diabetes.png
graph.write_png('Shool.png')
Image(graph.create_png())
#import matplotlib.pyplot as plt
#import cv2
#%matplotlib inline
#img = cv2.imread('diabetes.png')
#plt.figure(figsize = (20, 20))
#plt.imshow(img)


# In[28]:





# In[ ]:





# In[ ]:





# In[ ]:




