#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

df = pd.read_csv('/home/minhvu/Desktop/Py4DS/cuối kì/2020/bank_marketing/bank-full.csv')
df.head()


# #**B. Processing data**

# ##**1. Pre-processing data**

# In[2]:


df.info()


# **Loai gia tri trung**

# In[3]:


N = len(df)  # Count the number of rows in data
print(N)     

df.drop_duplicates(inplace = True) # df after dropping duplicates
print("The new dimension after checking duplicate & removing is:\t (%s, %s)"%(df.shape)) #size of data (rows,columns)
print('There are %s observations is duplicated, take %s percentage on total dataset'%(N - len(df), 
                                                                                      round(100*(N - len(df))/N, 2)))


# **Remove outliers (Loại bỏ các giá trị ngoại lai))**

# In[4]:


### Tìm IQR của dữ liệu
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
## Loại bỏ outlier
outlier_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
df = df[~outlier_condition]
df.shape


# **Drop column according to NAN percentage for dataframe**

# In[5]:


df=df = df.loc[:, df.isnull().mean() < .8] # remove all columns has values which be NAN accounting for >= 80%

df.shape


# #**2. EDA** (Explotary Data Analysis)

# **Summary-statistic**

# In[6]:


df.describe()


# In[7]:


# Set 'y' = 'deposit' (y = {No_deposit, Yes_deposit})

# Build a function to show categorical values disribution
def plot_bar(column):
    # temp df 
    temp_1 = pd.DataFrame()
    # count categorical values
    temp_1['No_deposit'] = df[df['y'] == 'no'][column].value_counts()
    temp_1['Yes_deposit'] = df[df['y'] == 'yes'][column].value_counts()
    temp_1.plot(kind='bar')
    plt.xlabel(f'{column}')
    plt.ylabel('Number of clients')
    plt.title('Distribution of {} and deposit'.format(column))
    plt.show();


# In[8]:


# Build a function to show categorical values disribution
def plot_bar_stacked(column):
    # temp df 
    temp_1 = pd.DataFrame()
    # count categorical values
    temp_1['Open Deposit'] = df[df['y'] == 'yes'][column].value_counts()/(df[column].value_counts())
    temp_1.plot(kind='bar')
    plt.xlabel(f'{column}')
    plt.ylabel('Reponse Rate %')
    plt.title('Reponse Rate on offer'.format(column))
    plt.show();


# **Bar**

# In[9]:


plot_bar('job'), plot_bar_stacked('job')


# In[10]:


plot_bar('marital'), plot_bar_stacked('marital')


# In[11]:


plot_bar('education'), plot_bar_stacked('education')


# In[12]:


plot_bar('contact'), plot_bar_stacked('contact')


# In[13]:


plot_bar('poutcome'), plot_bar_stacked('poutcome')


# In[14]:


plot_bar('loan'), plot_bar_stacked('loan')


# In[15]:


plot_bar('housing'), plot_bar_stacked('housing')


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14);


# #**C. Train-Test**

# **Xử lý các dữ liệu category**

# In[17]:


df= df.drop(columns=['day','month'],axis = 1)


# In[18]:


cleaned_data  = pd.get_dummies(df)
cleaned_data.shape


# In[19]:


from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
df.head()


# **Select features**

# In[20]:


y = df['y']
X = df[['duration','pdays','previous','poutcome','housing','contact','age','balance','campaign','marital']]


# **Train - Test**

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# #**D. Choose model**

# ##**2. Model: Logistic Regression**

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = LogisticRegression(max_iter = len(y_train))
clf.fit(X_train,y_train)

print("Logistic Regression")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("test acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print("Random Forest Classifier")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("test acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))


# In[ ]:





# In[ ]:




