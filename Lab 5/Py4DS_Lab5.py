import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

'''
Function: impute_mode_age
Purpose: Impute missing value with mode(seperate by sex)
Parameter: 
  df_in: Input DataFrame
Return:
  DataFrame after impute missing value
#'''

def impute_mode_age(df_in):
    from scipy import stats
    a = np.array(df_in.groupby(['Sex'])['Age'])
    female_mode = int(stats.mode(a[0][1])[0])
    male_mode = int(stats.mode(a[1][1])[0])
    df_in = df_in.reset_index(drop = True)
    male_df = df_in[df_in['Sex'] == 1]
    female_df = df_in[df_in['Sex'] == 0]
    male_df = male_df.fillna(male_mode)
    female_df = female_df.fillna(female_mode)
    return pd.concat([male_df,female_df]).sort_index()

'''
Function: impute_gbr_age
Purpose: Impute missing value using Gradient Boot Regression to predict
Parameter: 
  df_in: Input DataFrame
Return:
  DataFrame after impute missing value
#'''

def impute_gbr_age(df_in):
    from sklearn.ensemble import GradientBoostingRegressor
    test = df_in[df_in['Age']!=df_in['Age']]
    train = df_in[df_in['Age']==df_in['Age']]
    X_train = train.drop(['Age'],axis=1)
    y_train = train['Age']
    X_test = test.drop('Age',axis=1)
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    X_test['Age'] = y_pred
    return pd.concat([train,X_test]).sort_index()

'''
Function: impute_gbr_age
Purpose: Impute missing value using Linear Regression to predict
Parameter: 
  df_in: Input DataFrame
Return:
  DataFrame after impute missing value
#'''

def impute_linear_age(df):
    from sklearn.linear_model import LinearRegression
    test = df[df['Age']!=df['Age']]
    train = df[df['Age']==df['Age']]
    X_train = train.drop(['Age'],axis=1)
    y_train = train['Age']
    X_test = test.drop('Age',axis=1)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    X_test['Age'] = y_pred
    return pd.concat([train,X_test]).sort_index()

'''
Function: LE_columns
Purpose: Label Encoder columns with categorical data
Parameter: 
  df_in: Input DataFrame
Return:
  DataFrame after encoder
#'''

def LE_columns(df):
    columns = ['Name','Sex','Cabin','Embarked']
    for col in columns:
        if (not isinstance(df[col],(int,float))):
            df[col] = LabelEncoder().fit_transform(df[col])

'''
Function: balancing_data
Purpose: Handle imbalanced data
Parameter: 
  df_in: Input DataFrame
Return:
  DataFrame after handling imbalanced data
#'''

def balancing_data(X_train,Y_train):
    import imblearn
    from imblearn.over_sampling import SMOTE
    tmp = Y_train.value_counts(normalize=True)
  #if (tmp.min()>0.2):
    #return
    sm = SMOTE(random_state=2)
    X_train, Y_train = sm.fit_sample(X_train, Y_train.ravel())

'''
Function: RFCaccuracy
Purpose: Find Random Forest model's f1 score
Parameter: 
  df_in: Input DataFrame
Return:
  F1 score
#'''

def RFCaccuracy(X_train,X_test,Y_train,Y_test = []):
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    end = time.time()
    if (not Y_test==[]):
        score = f1_score(Y_test,Y_predict)
        print("Train set f1 score for Random Forest Classifier: ", f1_score(Y_train,clf.predict(X_train)))
        print("Test set f1 score for Random Forest Classifier: ", score)
        print("Time estimated: ",end-start)
        return score
    else:
        return Y_predict

'''
Function: LRaccuracy
Purpose: Find Logistic Regression model's f1 score
Parameter: 
  df_in: Input DataFrame
Return:
  F1 score
#'''

def LRaccuracy(X_train,X_test,Y_train,Y_test):
    import time
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    start = time.time()
    clf = LogisticRegression(max_iter = X_train.shape[0]).fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    score = f1_score(Y_test,Y_predict)
    end = time.time()
    print("Train set f1 score for Logistic Regression: ", f1_score(Y_train,clf.predict(X_train)))
    print("Test set f1 score for Logistic Regression: ", score)
    print("Time estimated: ",end-start)
    return score

'''
Function: SVCaccuracy
Purpose: Find Suport Vector Machine model's f1 score
Parameter: 
  df_in: Input DataFrame
Return:
  F1 score
#'''

def SVCaccuracy(X_train,X_test,Y_train,Y_test):
    import time
    from sklearn import svm
    from sklearn.metrics import f1_score
    start = time.time()
    clf = svm.SVC()
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    score = f1_score(Y_test,Y_predict)
    end = time.time()
    print("Train set f1 score for Support Vector Classification: ", f1_score(Y_train,clf.predict(X_train)))
    print("Test set f1 score for Support Vector Classification: ", score)
    print("Time estimated: ",end-start)
    return score

'''
Function: DTCaccuracy
Purpose: Find Decision Tree model's f1 score
Parameter: 
  df_in: Input DataFrame
Return:
  F1 score
#'''

def DTCaccuracy(X_train, X_test,Y_train,Y_test = []):
    import time
    from sklearn import tree
    from sklearn.metrics import f1_score
    start = time.time()
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    score = f1_score(Y_test,Y_predict)
    end = time.time()
    print("Train set f1 score for Decision Tree Classification: ", f1_score(Y_train,clf.predict(X_train)))
    print("Test set f1 score for Decision Tree Classification: ", score)
    print("Time estimatimated: ",end-start)
    return score

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
    #print("There are {0} values have been imputed".format(count_missing))
    return df_out

'''
Function: feature_engine
Purpose: Handle feature(Create feature, Delete feature, Label encoding, Handle 
         missing data)
Parameter: 
  df_in: Input DataFrame
Return:
  DataFrame after handle feature
'''

def feature_engine(df_in):
    df_out = df_in.copy()
    df_out = df_out.drop(['PassengerId','Ticket'],axis = 1)
    df_out['FamilySize'] = df_out['SibSp']+df_out['Parch']
    df_out['FarePerPerson'] = df_out['Fare']/(df_out['FamilySize']+1)
    df_out = df_out.drop(['Fare'],axis = 1)
    df_out['Cabin'] = np.where(df_out['Cabin']!=df_out['Cabin'],' ',df_out['Cabin'])
    df_out['Cabin'] = [s[0] for s in df_out['Cabin']]
    df_out['Name'] = [re.findall(', ([^ ]*). ',s)[0] for s in df_out['Name']]
    df_out = impute_mode_value(df_out, df_out.columns.drop("Age"))
    LE_columns(df_out)
    '''
    for i in df_out.columns:
        print(df_out[i].value_counts())
        print(df_out[i].isnull().sum())
        print("-"*30)
    #'''
    #df_out = impute_mode_age(df_out)
    df_out = impute_gbr_age(df_out)
    #df_out = impute_linear_age(df_out)
    return df_out

def main():
    path = '/home/minhvu/Documents/Py4DS_Lab5/Py4DS_Lab5_Dataset/titanic_train.csv'
    df_raw = pd.read_csv(path)
    df = feature_engine(df_raw)
    '''
    # Test code
    X = df.drop(['Survived'],axis = 1)
    Y = df['Survived']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2))
    balancing_data(X_train,Y_train)
    return RFCaccuracy(X_train,X_test,Y_train,Y_test)
    #'''
    # Product
    path_test = '/home/minhvu/Documents/Py4DS_Lab5/Py4DS_Lab5_Dataset/titanic_test.csv'
    df_test_raw = pd.read_csv(path_test)
    df_test = feature_engine(df_test_raw)
    X_train = df.drop(['Survived'],axis = 1)
    Y_train = df['Survived']
    X_test = df_test
    Y_pred = RFCaccuracy(X_train,X_test,Y_train)
    df_test_raw['Survived'] = Y_pred
    path_predict = '/home/minhvu/Documents/Py4DS_Lab5/Py4DS_Lab5_Dataset/titanic_predict.csv'
    df_test_raw.to_csv(path_predict,index = False)
    display(df_test_raw)
  #'''

main()
