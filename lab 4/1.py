#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


'''
Purpose: Count missing values and columns
Parameters: 
    df_in: Dataframe input need to detect missing value
Returns:
    None
#'''

def detect_missing_values(df_in):
    df_miss = df_in.isnull().sum()
    #print(df_miss)
    df_miss = df_miss[df_miss>0]
    print("There are {0} columns have missing values:\n{1}".format(len(df_miss),df_miss))


# In[3]:


'''
Purpose: Drop missing value
Parameters: 
    df_in: Dataframe input need to drop
    list_columns: List of columns need to detect and drop missing value
Returns:
    Dataframe after droping missing value
#'''

def drop_missing_values(df_in, list_columns):
    df_out = df_in.copy()
    #print(df_out)
    #print(list_columns)
    count_missing = 0
    for column in list_columns:
        try:
            count_missing = count_missing + df_out[column].isnull().sum()
            #print("Remove {0} missing values from {1} column".format(df_out[column].isnull().sum(),column))
            df_out = df_out[~df_out[column].isnull()].copy()
        except:
            print("Some thing error with column {0}".format(column))
    print("There are {0} values have been removed".format(count_missing))
    return df_out


# In[4]:


'''
Purpose: Replace null values with most frequency value
Parameters: 
    df_in: Dataframe input
    list_columns: List of columns need to detect missing value and replace
Returns:
    Dataframe after replace null values with most frequency value
#'''

def impute_mode_value(df_in, list_columns):
    df_out = df_in.copy()
    count_missing = 0
    for column in list_columns:
        try:
            count_missing = count_missing + df_out[column].isnull().sum()
            #print("Impute {0} missing values from {1} column".format(df_out[column].isnull().sum(),column))
            mode_value = df_out[column].mode()[0]
            df_out[column].fillna(mode_value,inplace = True)
        except:
            print("Some thing error with column {0}".format(column))
    print("There are {0} values have been imputed".format(count_missing))
    return df_out


# In[5]:


'''
Purpose: Drop feature with highly correlation
Parameters: 
    df_in: Dataframe input
Returns:
    df_in: If there ain't any highly correlation features
    Dataframe: If there are highly correlation features, return dataframe after 
    remove feature
#'''

def drop_highly_corr_feature(df_in):
    df_out = df_in.copy()
    corr = df_out.corr()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column])>0.95)]
    print("There are {0} highly correlation feature has been removed\n{1}".format(len(to_drop),to_drop))
    if (len(to_drop)==0):
        return df_in
    else:
        return df_out.drop(to_drop,axis=1)


# In[6]:


'''
Purpose: Drop feature with high null values
Parameters: 
    df_in: Dataframe input
    threshold: threshold to remove feature
Returns:
    Remove columns if percentage of nan greater than threshold. Return dataframe 
    after remove columns
#'''

def drop_high_nan_column(df_in, threshold = .8):
    res = df_in.loc[:,df_in.isnull().mean()<threshold]
    before = df_in.shape[1]
    after = res.shape[1]
    print("There are {0} columns have been removed because of high nan percentage".format(before-after))
    return res


# In[7]:


'''
Purpose: Remove outlier
Parameters: 
    df_in: Dataframe input
Returns:
    Dataframe after remove outlier
#'''

def remove_outliers(df, out_cols, T=1.5, verbose=True):
    # Copy of df
    new_df = df.copy()
    init_shape = new_df.shape
    # For each column
    for i in out_cols:
        q1 = new_df[i].quantile(.25)
        q3 = new_df[i].quantile(.75)
        col_iqr = q3 - q1
        col_max = q3 + T * col_iqr
        col_min = q1 - T * col_iqr
        # Filter data without outliers and ignoring nan
        filtered_df = new_df[(new_df[i] <= col_max) & (new_df[i] >= col_min)]
        if verbose:
            n_out = new_df.shape[0] - filtered_df.shape[0] 
        new_df = filtered_df
            
    if verbose:
        # Print shrink percentage
        lines_red = df.shape[0] - new_df.shape[0]
    return new_df


# In[8]:


'''
Purpose: Generate next permutation
Parameters: 
    a: A state
Returns:
    Next state of a
#'''

def next_permutation(a):
    for i in range(len(a)-2,-1,-1):
        if (a[i]<a[i+1]):
            break
        if (a[i]>a[i+1]):
            return False
    for j in range(len(a)-1,-1,-1):
        if (a[j]>a[i]):
            break
    a[i], a[j] = a[j], a[i]
    a[i+1:] = reversed(a[i+1:])
    return True


# In[9]:


'''
Purpose: Find label to transform y_pred to maximize accuracy with y_true
Parameters: 
    y_true: Array of true label 
    y_pred: Array of predict label
Returns:
    Best label to tranform y_pred
#'''

def find_best_label(y_true, y_pred):
    y_temp = y_true.copy()
    yp_temp = y_pred.copy()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unique = np.unique(y_true)
    #*********************************
    #print(y_true)
    #print(y_pred)
    #*********************************
    label = []
    for i in range(0,len(unique)):
        label.append(i)
    max_count = 0
    best_label = []
    while (True):
        counts = 0
        for i in range(0,len(y_true)):
            if (label[y_pred[i]] == y_true[i]):
                counts = counts + 1
        if (counts>max_count):
            max_count = counts
            best_label = label.copy()
        if (not next_permutation(label)):
            y_true = y_temp
            return best_label
    y_true = y_temp
    y_pred = yp_temp
    return best_label


# In[10]:


'''
Purpose: Transform y to a new vector with label
Parameters: 
    y: Array of label
    label: Label to transform
Returns:
    Vector after transform
#'''

def change_label(y,label):
    res = []
    for i in range(0,len(y)):
        res.append(label[y[i]])
    return res


# In[11]:


'''
Purpose: Train data with Kmeans model then print accuracy
Parameters: 
    X_train: Train set feature
    y_train: Train set label
    X_test: Test set feature
    y_test: Test se label
Returns:
    None
#'''

def KMeans_model(X_train, y_train, X_test, y_test):
    unique = np.unique(y_train)
    cls = KMeans(n_clusters = len(unique), max_iter = 
len(y_train),random_state=42)
    cls = cls.fit(X_train)
    y_pred_train = np.array(cls.labels_)
    label = find_best_label(y_train,y_pred_train)
    y_pred_train = change_label(y_pred_train,label)
    y_pred = cls.predict(X_test)
    y_pred = change_label(y_pred,label)
    print("Accuracy train: ",accuracy_score(y_train,y_pred_train))
    print("Accuracy test: ", accuracy_score(y_test,y_pred))
    print("Confusion_matrix: ", confusion_matrix(y_test,y_pred))


# In[12]:


'''
Purpose: Train data with Gaussian Mixture model then print accuracy
Parameters: 
    X_train: Train set feature
    y_train: Train set label
    X_test: Test set feature
    y_test: Test se label
Returns:
    None
#'''

def GaussianMixture_model(X_train, y_train, X_test, y_test):
    unique = np.unique(y_train)
    cls = GaussianMixture(n_components= len(unique), max_iter = len(y_train),random_state=42)
    cls = cls.fit(X_train)
    y_pred_train = np.array(cls.predict(X_train))
    label = find_best_label(y_train,y_pred_train)
    y_pred_train = change_label(y_pred_train,label)
    y_pred = cls.predict(X_test)
    y_pred = change_label(y_pred,label)
    print("Accuracy train: ",accuracy_score(y_train,y_pred_train))
    print("Accuracy test: ", accuracy_score(y_test,y_pred))
    print("Confusion_matrix: ", confusion_matrix(y_test,y_pred))


# In[13]:


'''
Purpose: Train data with Agglomerative Clustering model then print accuracy
Parameters: 
    X_train: Train set feature
    y_train: Train set label
    X_test: Test set feature
    y_test: Test se label
Returns:
  None
#'''

def AgglomerativeClustering_model(X_train, y_train, X_test, y_test):
    unique = np.unique(y_train)
    cls = AgglomerativeClustering(n_clusters= len(unique))
    cls = cls.fit(X_train)
    y_pred_train = np.array(cls.labels_)
    label = find_best_label(y_train,y_pred_train)
    y_pred_train = change_label(y_pred_train,label)
    y_pred = cls.fit_predict(X_test)
    y_pred = change_label(y_pred,label)
    print("Accuracy train: ",accuracy_score(y_train,y_pred_train))
    print("Accuracy test: ", accuracy_score(y_test,y_pred))
    print("Confusion_matrix: ", confusion_matrix(y_test,y_pred))


# In[14]:


def RandomForestClassifier_model(X, y, X_train, y_train, X_test, y_test):
    #Fit Random Forest Classifier
    rdf=RandomForestClassifier(max_depth=2, random_state=0)
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


# In[15]:


def LogisticRegressionClassifier_model(X, y, X_train, y_train, X_test, y_test):
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


# In[16]:


def main():
    path='/home/minhvu/Documents/Py4DS_Lab1_Dataset/spam.csv'
    #Read in the data into a pandas dataframe
    df=pd.read_csv(path)
    #Read in the data into a numpy array
    df=np.genfromtxt(path,delimiter=',')

    #df = pd.read_csv('/home/minhvu/Documents/Py4DS_Lab1_Dataset/spam.csv')
    #df = drop_highly_corr_feature(df)
    #df = remove_outliers(df,df.columns,T=1.5, verbose=True)
    #df = drop_high_nan_column(df,0.8)
    #df = impute_mode_value(df,df.columns)
    #df = drop_missing_values(df,df.columns)
    #for col in df.columns:
     #   print("*"*30)
      #  print(df[col].describe())
       # df[col] = LabelEncoder().fit_transform(df[col])
    #print(df.dtypes)
    X=df[:,0:len(df[0])-1]
    y=df[:,len(df[0])-1]
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state=42)
    print("*"*40)
    print("KMeans clustering model: ")
    KMeans_model(X_train, y_train, X_test, y_test)
    print("*"*40)
    print("Gaussian Mixture clustering model: ")
    GaussianMixture_model(X_train, y_train, X_test, y_test)
    print("*"*40)
    print("Agglomerative Clustering model: ")
    AgglomerativeClustering_model(X_train, y_train, X_test, y_test)
    print("*"*40)
    print("So sánh với kết quả bài hôm trước")
    print("Random Forest Classifier_model")
    RandomForestClassifier_model(X, y, X_train, y_train, X_test, y_test)
    print("*"*40)
    print("Logistic Regression Classifier model")
    LogisticRegressionClassifier_model(X, y, X_train, y_train, X_test, y_test)


# In[17]:


main()

