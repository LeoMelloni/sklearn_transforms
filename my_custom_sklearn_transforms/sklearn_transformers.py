!pip install scikit-learn==0.20.0 --upgrade
import json

import requests

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline


from sklearn.model_selection import KFold, cross_validate

import pandas as pd

uri = "https://github.com/maratonadev-br/desafio-2-2020/blob/master/Assets/Data/dataset_desafio_2.csv?raw=true%27"
df_data_1 = pd.read_csv(uri)

df_data_1.head()

df_data_1.info()

df_data_1.describe()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 4))

sns.countplot(ax=axes[0], x='REPROVACOES_DE', data=df_data_1)
sns.countplot(ax=axes[1], x='REPROVACOES_EM', data=df_data_1)
sns.countplot(ax=axes[2], x='REPROVACOES_MF', data=df_data_1)
sns.countplot(ax=axes[3], x='REPROVACOES_GO', data=df_data_1)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 4))

sns.distplot(df_data_1['NOTA_DE'], ax=axes[0])
sns.distplot(df_data_1['NOTA_EM'], ax=axes[1])
sns.distplot(df_data_1['NOTA_MF'], ax=axes[2])
sns.distplot(df_data_1['NOTA_GO'].dropna(), ax=axes[3])

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 4))

sns.countplot(ax=axes[0], x='INGLES', data=df_data_1)
sns.countplot(ax=axes[1], x='FALTAS', data=df_data_1)
sns.countplot(ax=axes[2], x='H_AULA_PRES', data=df_data_1)
sns.countplot(ax=axes[3], x='TAREFAS_ONLINE', data=df_data_1)

fig = plt.plot()
sns.countplot(x='PERFIL', data=df_data_1)

from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        return data.drop(labels=self.columns, axis='columns')
    
    rm_columns = DropColumns(
    columns=["NOME", "MATRICULA", "H_AULA_PRES", ]  
)

print(rm_columns)

print(df_data_1.columns)

rm_columns.fit(X=df_data_1)

df_data_2 = pd.DataFrame.from_records(
    data=rm_columns.transform(
        X=df_data_1
    ),
)

print("Colunas do dataset após a transformação ``DropColumns``: \n")
print(df_data_2.columns)

si = SimpleImputer(
    missing_values=np.nan,  
    strategy='constant',  
    fill_value=0,  
    verbose=0,
    copy=True
)

print("Valores nulos antes da transformação SimpleImputer: \n\n{}\n".format(df_data_2.isnull().sum(axis = 0)))

si.fit(X=df_data_2)


df_data_3 = pd.DataFrame.from_records(
    data=si.transform(
        X=df_data_2
    ),  
    columns=df_data_2.columns  
)

print("Valores nulos no dataset após a transformação SimpleImputer: \n\n{}\n".format(df_data_3.isnull().sum(axis = 0)))

df_data_3.head()

features = [
    "MATRICULA", 'REPROVACOES_DE', 'REPROVACOES_EM', "REPROVACOES_MF", "REPROVACOES_GO",
    "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO",
    "INGLES", "H_AULA_PRES", "TAREFAS_ONLINE", "FALTAS", 
]

target = ["PERFIL"]

X = df_data_1[features]
y = df_data_1[target]

X.head()

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=337)

from xgboost import XGBClassifier

xgboost = XGBClassifier(learning_rate = 0.1, booster='gbtree', min_child_weight=0.1)
xgboost.fit(
    X_train,
    y_train
)

y_pred = xgboost.predict(X_test)

X_test.head()

from sklearn.metrics import accuracy_score

print("Acurácia: {}%".format(100*round(accuracy_score(y_test, y_pred), 4)))

