import sklearn
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

train_y=df_train['hasPaid']
code=df_test['code']
test_y=df_test['hasPaid']

df_train.drop(['code', 'hasPaid'], axis=1, inplace=True)
df_test.drop(['code', 'hasPaid'], axis=1, inplace=True)

#one-hot encoding
all_data = pd.concat((df_train,df_test))
for column in all_data.select_dtypes(include=[np.object]).columns:
    df_train[column] = df_train[column].astype('category', categories = all_data[column].unique())
    df_test[column] = df_test[column].astype('category', categories = all_data[column].unique())

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

#feature scaling
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
df_train = min_max_scaler.fit_transform(df_train)
df_test = min_max_scaler.fit_transform(df_test)

#PCA
pca = PCA(0.82)
pca.fit(df_train)
df_train = pca.transform(df_train)
df_test= pca.transform(df_test)

#PCA plot
per_var= np.round(pca.explained_variance_ratio_*1500,decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('percentage of explained label')
plt.xlabel('principal component')
plt.title('Screen plot')
plt.show()

#training 
svm=SVC(gamma=0.25, C=0.82)
svm.fit(df_train, train_y)

#prediction
y_train_predicted=svm.predict(df_train)
y_test_predicted=svm.predict(df_test)
    
print(sklearn.metrics.f1_score(train_y, y_train_predicted, labels=None, pos_label=1, average="micro", sample_weight=None))
print(sklearn.metrics.f1_score(test_y.astype(int), y_test_predicted.astype(int), labels=None, pos_label=1, average="micro", sample_weight=None))
