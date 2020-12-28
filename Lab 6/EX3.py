import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import re
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def Constant_Features(x_train, x_test,threshold=0):
    """
    Removing Constant Features using Variance Threshold
    Input: threshold parameter to identify the variable as constant
         train data (pd.Dataframe) 
         test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # import and create the VarianceThreshold object.
    from sklearn.feature_selection import VarianceThreshold
    vs_constant = VarianceThreshold(threshold)

    # select the numerical columns only.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]

    # fit the object to our data.
    vs_constant.fit(numerical_x_train)

    # get the constant colum names.
    constant_columns = [column for column in numerical_x_train.columns
                      if column not in numerical_x_train.columns[vs_constant.get_support()]]

    # detect constant categorical variables.
    constant_cat_columns = [column for column in x_train.columns 
                          if (x_train[column].dtype == "O" and len(x_train[column].unique())  == 1 )]

    # concatenating the two lists.
    all_constant_columns = constant_cat_columns + constant_columns
  
    return all_constant_columns

def Quansi_Constant_Feature(x_train, x_test,threshold=0.98):
    # create empty list
    quasi_constant_feature = []

    # loop over all the columns
    for feature in x_train.columns:
      # calculate the ratio.
        predominant = (x_train[feature].value_counts() / np.float(len(x_train))).sort_values(ascending=False).values[0]
    
      # append the column name if it is bigger than the threshold
        if predominant >= threshold:
            quasi_constant_feature.append(feature)   
    return quasi_constant_feature

def Dupplicate_Feature(x_train,x_test):
    # transpose the feature matrice
    train_features_T = x_train.T

    # print the number of duplicated features
    print(train_features_T.duplicated().sum())

    # select the duplicated features columns names
    duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

    return duplicated_columns

def Correlated_Feature(x_train,x_test,threshold=0.8):
    # creating set to hold the correlated features
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = x_train.corr()
    '''
    # optional: display a heatmap of the correlation matrix
    plt.figure(figsize=(11,11))
    sns.heatmap(corr_matrix)
    #'''

    for i in range(len(corr_matrix .columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
    return corr_features

def Mutual_Information(x_train, x_test, select_k = 10):
    # import the required functions and object.
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest

    # get only the numerical features.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]


    # create the SelectKBest with the mutual info strategy.
    selection = SelectKBest(mutual_info_classif, k=select_k)
    selection.fit(numerical_x_train, y_train)

    # display the retained features.
    features = x_train.columns[selection.get_support()]
    return features

def Select_Model(x_train,y_train,x_test,y_test):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    # feature extraction
    select_model = SelectFromModel(rfc)
    # fit on train set
    fit = select_model.fit(x_train, y_train)
    # transform train set
    x_train = fit.transform(x_train)
    x_test = fit.transform(x_test)
    return x_train, x_test

def PCA_Feature(x_train,x_test):
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_train = pca.transform(x_train)

def RFE_Feature(x_train,y_train,x_test,y_test):
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    rfe = RFE(estimator=rfc, n_features_to_select=3)
    # fit the model
    rfe.fit(x_train, y_train)
    #transform the data
    x_train, y_train = rfe.transform(x_train, y_train)
    x_test, y_test = rfe.transform(x_test, y_test)
    return x_train,y_train,x_test,y_test

def Drop_Columns(x_train,x_test,columns):
    x_train = x_train.drop(labels=columns, axis=1, inplace=True)
    x_test = x_test.drop(labels=columns, axis=1, inplace=True)
    return x_train, x_test

def RFCaccuracy(X_train,X_test,Y_train,Y_test):
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    score = accuracy_score(Y_test,Y_predict)
    end = time.time()
    print("Accuracy score for Random Forest Classifier: ", score)
    print("Time estimated: ",end-start)
    return score

path='/home/minhvu/Documents/Py4DS_Lab6/Py4DS_Lab6_Dataset/data.csv'
df = pd.read_csv(path)
df

"""- Dropping features"""

df.columns

df=df.drop(['Unnamed: 0','ID','Name','Photo','Flag','Club Logo','Joined'],axis = 1)
df=df.drop(['Loaned From','Contract Valid Until','Real Face'],axis = 1)
df

df.select_dtypes(include=['object']).columns

"""- Observation:
  + We have 3 price columns: Value, Wage, Release Clause
  + Price columns have 3 type of unit: M(Million) ,K(Thousand), 0(0 euro)
"""

list_money = ['Value','Wage','Release Clause']
for col in list_money:
    print(col)
    unit = [str[len(str)-1] if (str==str) else str for str in df[col]]
    unique, count = np.unique(unit,return_counts=True)
    fig = plt.figure(figsize=(10,7))
    plt.pie(count,labels=unique)
    print(count)

for col in list_money:
    number = [s[1:len(s)-1] if (s==s) else s for s in df[col]]
    number = ['0' if (s=='') else s for s in number]
    number = np.array(number).astype(np.float)
    unit = [str[len(str)-1] if (str==str) else str for str in df[col]]
    for i in range(len(unit)):
        if (unit[i]=='M'):
            number[i] = number[i]*1000000
        elif (unit[i]=='K'):
            number[i] = number[i]*1000
    df[col] = number
    print(number)
  #'''
df

df.select_dtypes(include=['object']).columns

"""- Obsevation
  + Height columns has format 5'8
  + Weight columns has format 176lbs
"""

number = [s.replace('\'','.') if (s==s) else s for s in df['Height']]
df['Height'] = np.array(number).astype(np.float)
df

number = [s[0:len(s)-3] if (s==s) else s for s in df['Weight']]
df['Weight'] = np.array(number).astype(np.float)
df

"""- Observation:
  + Some columns has format 34+2
- Decision:
  + Get number before add notation
"""

list_position = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM',
       'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM',
       'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for col in list_position:
    number = [re.findall('([0-9]+)\+',s)[0] if (s==s) else s for s in df[col]]
    df[col] = np.array(number).astype(np.float)
df

"""- Observation:
  + Work Rate has format Medium/ Medium or something like */* with * in Low, Medium or High
- Decision:
  + Split to Work Rate 1 and Work Rate2
  + Delete Work Rate columns
"""

df.select_dtypes(include=['object']).columns

s = [re.findall('([A-Z][a-z]+)',s) if (s==s) else s for s in df['Work Rate']]
df['Work Rate 1'] = [s1[0] if (s1==s1) else s1 for s1 in s]
df['Work Rate 2'] = [s2[1] if (s2==s2) else s2 for s2 in s]
df = df.drop(['Work Rate'],axis=1)
df



list_category = df.select_dtypes(include=['object']).columns
for col in list_category:
    df_not_nan = df[col][pd.notnull(df[col])]
    print(col)
    unique, count = np.unique(df_not_nan,return_counts=True)
    fig = plt.figure(figsize=(10,7))
    plt.pie(count,labels=unique)

df['Body Type'] = [s if (s in ['Lean','Normal','Stocky']) else 'Other'for s in df['Body Type']]
df

"""- Feature Engineering"""

df.shape

df.info()

list_onehot = ['Preferred Foot','Work Rate 1','Work Rate 2','Body Type']
df = pd.get_dummies(df,columns = list_onehot)
df

le = preprocessing.LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    print(col)
    df_tmp = df[pd.notnull(df[col])]
    df_tmp[col] = le.fit_transform(df_tmp[col])

'''
Purpose: Replace null values with most frequency value
Parameters: 
  df_in: Dataframe input
  list_columns: List of columns need to detect missing value and replace
Returns:
  Dataframe after replace null values with most frequency value
#'''

def impute_mode_value(X_train, X_test, list_columns):
    count_missing = 0
    for column in list_columns:
        if (X_train[column].isna().sum()==0):
            continue
        try:
            count_missing = count_missing + X_train[column].isnull().sum() + X_test[column].isnull().sum()
            #print("Impute {0} missing values from {1} column".format(df_out[column].isnull().sum(),column))
            mode_value = X_train[column].mode()[0]
            X_train[column].fillna(mode_value,inplace = True)
            X_test[column].fillna(mode_value,inplace = True)
        except:
            print("Some thing error with column {0}".format(column))
    print("There are {0} values have been imputed".format(count_missing))
    return X_train, X_test

def label_encoder(x_train, x_test, list_columns):
    for col in list_columns:
        le = preprocessing.LabelEncoder()
        le.fit(x_train[col])
        x_train[col] = le.transform(x_train[col])
        x_test[col] = le.fit_transform(x_test[col])
    return x_train, x_test

df = df[pd.notnull(df['Position'])]
X = df.drop(['Position'],axis=1)
y = df['Position']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

le_test = preprocessing.LabelEncoder()
le_test.fit(y_train)
y_train = le_test.transform(y_train)
y_test = le_test.transform(y_test)

impute_mode_value(x_train, x_test,x_train.columns)

lst_obj = x_train.select_dtypes(include = ['object']).columns
label_encoder(x_train,x_test,lst_obj)

x_train.info()

x_test.info()

def Constant_Features(x_train, x_test,threshold=0):
    """
    Removing Constant Features using Variance Threshold
    Input: threshold parameter to identify the variable as constant
             train data (pd.Dataframe) 
             test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
      # import and create the VarianceThreshold object.
    from sklearn.feature_selection import VarianceThreshold
    vs_constant = VarianceThreshold(threshold)

      # select the numerical columns only.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]

      # fit the object to our data.
    vs_constant.fit(numerical_x_train)

      # get the constant colum names.
    constant_columns = [column for column in numerical_x_train.columns
                          if column not in numerical_x_train.columns[vs_constant.get_support()]]

      # detect constant categorical variables.
    constant_cat_columns = [column for column in x_train.columns 
                              if (x_train[column].dtype == "O" and len(x_train[column].unique())  == 1 )]

      # concatenating the two lists.
    all_constant_columns = constant_cat_columns + constant_columns

    return all_constant_columns

def Quansi_Constant_Feature(x_train, x_test,threshold=0.98):
  # create empty list
    quasi_constant_feature = []

  # loop over all the columns
    for feature in x_train.columns:
    # calculate the ratio.
        predominant = (x_train[feature].value_counts() / np.float(len(x_train))).sort_values(ascending=False).values[0]
    
    # append the column name if it is bigger than the threshold
        if predominant >= threshold:
            quasi_constant_feature.append(feature)   
    return quasi_constant_feature

def Dupplicate_Feature(x_train,x_test):
    # transpose the feature matrice
    train_features_T = x_train.T

    # print the number of duplicated features
    print(train_features_T.duplicated().sum())

    # select the duplicated features columns names
    duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

    return duplicated_columns

def Correlated_Feature(x_train,x_test,threshold=0.8):
    # creating set to hold the correlated features
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = x_train.corr()
    '''
    # optional: display a heatmap of the correlation matrix
    plt.figure(figsize=(11,11))
    sns.heatmap(corr_matrix)
    #'''

    for i in range(len(corr_matrix .columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
    return corr_features

def Mutual_Information(x_train, x_test, select_k = 10):
    # import the required functions and object.
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest

    # get only the numerical features.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]


    # create the SelectKBest with the mutual info strategy.
    selection = SelectKBest(mutual_info_classif, k=select_k)
    selection.fit(numerical_x_train, y_train)

    # display the retained features.
    features = x_train.columns[selection.get_support()]
    return features

def Select_Model(x_train,y_train,x_test,y_test):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    # feature extraction
    select_model = SelectFromModel(rfc)
    # fit on train set
    fit = select_model.fit(x_train, y_train)
    # transform train set
    x_train = fit.transform(x_train)
    x_test = fit.transform(x_test)
    return x_train, x_test

def PCA_Feature(x_train,x_test):
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_train = pca.transform(x_train)

def RFE_Feature(x_train,y_train,x_test,y_test):
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    rfe = RFE(estimator=rfc, n_features_to_select=3)
    # fit the model
    rfe.fit(x_train, y_train)
    #transform the data
    x_train, y_train = rfe.transform(x_train, y_train)
    x_test, y_test = rfe.transform(x_test, y_test)
    return x_train,y_train,x_test,y_test

def Drop_Columns(x_train,x_test,columns):
    x_train = x_train.drop(labels=columns, axis=1, inplace=True)
    x_test = x_test.drop(labels=columns, axis=1, inplace=True)
    return x_train, x_test

def RFCaccuracy(X_train,X_test,Y_train,Y_test):
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    score = accuracy_score(Y_test,Y_predict)
    end = time.time()
    print("Accuracy score for Random Forest Classifier: ", score)
    print("Time estimated: ",end-start)
    return score

cf = Constant_Features(x_train, x_test, threshold=0)
Drop_Columns(x_train,x_test,cf)
print(cf)
#'''

qcf = Quansi_Constant_Feature(x_train,y_train)
print(qcf)
Drop_Columns(x_train,x_test,qcf)
#'''

d_f = Dupplicate_Feature(x_train, x_test)
print(d_f)
Drop_Columns(x_train,x_test,d_f)
#'''

c_f = Correlated_Feature(x_train,x_test,threshold=0.8)
print(c_f)
Drop_Columns(x_train,x_test,c_f)
#'''

x_train_mi = x_train.copy()
y_train_mi = y_train.copy()
x_test_mi = x_test.copy()
y_test_mi = y_test.copy()
m_i = Mutual_Information(x_train_mi,x_test_mi, select_k = 10)
print(m_i)
x_train_mi = x_train_mi[m_i]
x_test_mi = x_test_mi[m_i]
#'''

x_train_sm = x_train.copy()
y_train_sm = y_train.copy()
x_test_sm = x_test.copy()
y_test_sm = y_test.copy()
Select_Model(x_train_sm,y_train_sm,x_test_sm,y_test_sm)
#'''

x_train_pca = x_train.copy()
y_train_pca = y_train.copy()
x_test_pca = x_test.copy()
y_test_pca = y_test.copy()
PCA_Feature(x_train_pca,y_train_pca)

'''
x_train_rfe = x_train.copy()
y_train_rfe = y_train.copy()
x_test_rfe = x_test.copy()
y_test_rfe = y_test.copy()
RFE_Feature(x_train_rfe,y_train_rfe,x_test_rfe,y_test_rfe)
#'''

print("Select Model")
RFCaccuracy(x_train_sm,x_test_sm,y_train_sm,y_test_sm)
print("*"*30)
print("Mutual Infomation")
RFCaccuracy(x_train_mi,x_test_mi,y_train_mi,y_test_mi)
print("*"*30)
print("PCA")
RFCaccuracy(x_train_pca,x_test_pca,y_train_pca,y_test_pca)
print("*"*30)

y.value_counts()