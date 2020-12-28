import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import eli5
#from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
#from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

path1='/home/minhvu/Documents/Py4DS_Lab6/Py4DS_Lab6_Dataset/train.csv'
path2='/home/minhvu/Documents/Py4DS_Lab6/Py4DS_Lab6_Dataset/test.csv'
train=pd.read_csv(path1)
test=pd.read_csv(path2)

labels = train.columns.drop(['id', 'target'])
train.shape

train.head()

test.head()

train[train.columns[2:]].std().plot(kind = 'kde')
plt.title('Standard Deviation')

train[train.columns[2:]].mean().plot(kind = 'kde')
plt.title('Mean')

train['target'].value_counts()

#Logistic Regression Model
sns.countplot(x = 'target', data = train, palette = 'hls')
plt.show
plt.savefig('count')

train.groupby('target').mean()

#logistic regression model
X = train.drop(['id','target'],axis = 1)
Y = train['target']
X_eval = test.drop(['id'], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)


model = LogisticRegression(solver = 'liblinear',C = 0.1, penalty = 'l1')
model.fit(x_train, y_train)
x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

print('----------------SMOTE--------------')
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', n_jobs=-1)
X_sm, y_sm = smote.fit_resample(X,Y)

df = pd.DataFrame(X_sm, columns = labels)
df['target'] = y_sm

sns.countplot(x = 'target', data = df, palette = 'hls')
plt.show
plt.savefig('count')

model = LogisticRegression(solver = 'liblinear',C = 1, penalty = 'l2')
normX = df.drop(['target'], axis = 1)
normY = df['target']

x_train, x_test, y_train, y_test = train_test_split(normX, normY, test_size=0.25, random_state=0)
model.fit(x_train, y_train)
x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')

rfe = RFE(model)
rfe.fit(X,Y)
print('selected features:')
print(labels[rfe.support_].tolist())

X_fs = rfe.transform(normX)
X_fs_eval = rfe.transform(X_eval)

model.fit(X_fs, normY)

pred = model.predict_proba(X_fs_eval)[:,1]
pred
