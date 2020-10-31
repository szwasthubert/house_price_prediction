import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
warnings.filterwarnings("ignore")

# %% READ DATA
df_train = pd.read_csv(os.getcwd()+"/analysis_data/train.csv")
df_train.drop(columns='Unnamed: 0', inplace=True)
X = df_train.drop(columns=['SalePrice'])
y = df_train['SalePrice'].to_numpy().reshape(-1, 1)

test_set = pd.read_csv("analysis_data/test.csv")
test_X = test_set.drop(['SalePrice', 'Unnamed: 0'], axis=1)
test_y = test_set['SalePrice'].to_numpy().reshape(-1, 1)

normalize = False
if normalize:
    cols_to_drop = ['Foundation', 'Neighborhood']
    cols_to_keep = [col for col in list(X.columns) if col not in cols_to_drop]

    X_cont = X.drop(columns=cols_to_drop)
    test_X_cont = test_X.drop(columns=cols_to_drop)
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X_cont)
    scaled_test_X = scaler.fit_transform(test_X_cont)

    test_X[cols_to_keep] = scaled_test_X
    X[cols_to_keep] = scaled_X

ensemble = pd.concat([X, test_X], axis=0)
ensemble = pd.get_dummies(ensemble)
X = ensemble.iloc[:X.shape[0], :]
test_X = ensemble.iloc[X.shape[0]:, :]

# %% TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    shuffle=True)

lr = LinearRegression()
lr.fit(X, y)
lr.score(X_test, y_test)
# %% TEST SET
svc.score(test_X, test_y)
