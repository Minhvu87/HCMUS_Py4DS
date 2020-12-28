#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv',header=None)


# In[3]:


data


# In[4]:


data[60].value_counts().plot(kind='barh')


# In[5]:


inputs_df = data.drop(60, axis =1)
inputs_df.head()


# In[6]:


targets_df = pd.get_dummies(data[60])
targets_df.head()


# In[7]:


rock_y_df = targets_df['R']
mine_y_df = targets_df['M']


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs_df,
                                                    mine_y_df, test_size=0.3, random_state=42)


# In[9]:


#Importing classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# In[10]:


classifiers_ = [
    ("AdaBoost",AdaBoostClassifier()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Linear SVM", SVC(kernel="linear", C=0.025,probability=True)),
    ("Naive Bayes",GaussianNB()),
    ("Nearest Neighbors",KNeighborsClassifier(3)),
    ("Neural Net",MLPClassifier(alpha=1)),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1)),
    ("RBF SVM",SVC(gamma=2, C=1,probability=True)),
    ("SGDClassifier", SGDClassifier(max_iter=1000, tol=10e-3,penalty='elasticnet'))
    ]


# In[11]:


clf_names = []
train_scores = []
test_scores = []
for n,clf in classifiers_:
    clf_names.append(n)
    # Model declaration with pipeline
    # For feature creation
    poly = PolynomialFeatures(2)
    clf = Pipeline([('POLY', poly),('CLF',clf)])
    
    # Model training
    clf.fit(X_train, y_train)
    print(n+" training done!")
    
    # Measure training accuracy and score
    train_scores.append(clf.score(X_train, y_train))
    print(n+" training score done!")
    
    # Measure test accuracy and score
    test_scores.append(clf.score(X_test, y_test))
    print(n+" testing score done!")
    print("---")


# In[12]:


#Plot results
plt.title('Accuracy Training Score')
plt.grid()
plt.plot(train_scores,clf_names)
plt.show()

plt.title('Accuraccy Test Score')
plt.grid()
plt.plot(test_scores,clf_names)
plt.show()


# In[13]:


rng = np.random.RandomState(1)
clf = GaussianProcessClassifier(1.0 * RBF(1.0))

clf = Pipeline([('POLY', poly),
                ('ADABOOST', clf)])
# Training our model
get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[14]:


from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(clf, X_train, y_train,
                             display_labels=['ROCK','MINE'],
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('Confusion matrix')

print('Train results: confusion matrix')
print(disp.confusion_matrix)


# In[15]:


from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score


# In[22]:





# In[23]:


disp = plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=['ROCK','MINE'],
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('Confusion matrix')

print('Test results: confusion matrix')
print(disp.confusion_matrix)


# In[24]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Recall: %.3f '%recall_score(
            y_true=y_test, y_pred=y_pred))
print('Precision: %.3f' %precision_score(
            y_true=y_test, y_pred=y_pred))
print('F1: %.3f'%recall_score(
            y_true=y_test, y_pred=y_pred))


# In[34]:


from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=10)
print('StratifiedKFold')
for train, test in skf.split(inputs_df, mine_y_df):
    print('train -  {}   |   test -  {}'.format(np.bincount(y_train), np.bincount(y_test)))

print('KFold')
kf = KFold(n_splits=10)
for train, test in kf.split(inputs_df, mine_y_df):
    print('train -  {}   |   test -  {}'.format(np.bincount(y_train), np.bincount(y_test)))

