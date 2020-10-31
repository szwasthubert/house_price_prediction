# %% CONFIG
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from command_files.CategoricalChooser import CustomCategoricalChooser
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import os
import math
from sklearn.model_selection import train_test_split
import scipy.stats as stats
warnings.filterwarnings("ignore")

# %% READ DATA
df = pd.read_csv(os.getcwd()+"/original_data/data.csv")

# %% NULL CHECK
df.isnull().sum().sort_values(ascending=False).to_csv("null.csv")

# %% DATA CLEANING. IMPUTATION
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

df.drop(columns=['GarageYrBlt'], inplace=True)
df.drop(columns=['MasVnrType', 'MasVnrArea'], inplace=True)

df = df[df["Electrical"].notna()]

df['PoolQC'] = df['PoolQC'].fillna("NP")  # NA -> NP == No Pool
df['MiscFeature'] = df['MiscFeature'].fillna("NMF")  # NA -> NMF == No MiscFeat
df['Alley'] = df['Alley'].fillna("NAA")  # NA -> NAA == No Alley Access
df['Fence'] = df['Fence'].fillna("NF")  # NA -> NAA == No Fence
df['FireplaceQu'] = df['FireplaceQu'].fillna("NFP")  # NA -> NAA == No Firep

df['GarageCond'] = df['GarageCond'].fillna("NG")  # NA -> NG == No Garage
df['GarageType'] = df['GarageType'].fillna("NG")  # NA -> NG == No Garage
df['GarageFinish'] = df['GarageFinish'].fillna("NG")  # NA -> NG == No Garage
df['GarageQual'] = df['GarageQual'].fillna("NG")  # NA -> NG == No Garage

df['BsmtExposure'] = df['BsmtExposure'].fillna("NB")  # NA -> NG == No Basement
df['BsmtQual'] = df['BsmtQual'].fillna("NB")  # NA -> NG == No Garage
df['BsmtCond'] = df['BsmtCond'].fillna("NB")  # NA -> NG == No Garage
df['BsmtFinType1'] = df['BsmtFinType1'].fillna("NB")  # NA -> NG == No Garage
df['BsmtFinType2'] = df['BsmtFinType2'].fillna("NB")  # NA -> NG == No Garage

df["LotFrontage"] = df["LotFrontage"].fillna(X_train["LotFrontage"].mean())  # Mean

X_train = df.iloc[:X_train.shape[0]].drop(columns=['SalePrice'])
X_test = df.iloc[X_train.shape[0]:].drop(columns=['SalePrice'])
y_train = df.iloc[:X_train.shape[0]]['SalePrice']
y_test = df.iloc[X_train.shape[0]:]['SalePrice']

# %% VISUALIZE
col = 'GrLivArea'
(mu, sigma) = stats.norm.fit(df[col])
sns.distplot(df[col], bins=100, kde=True)
plt.legend(["Dist: ($\mu=${0:.2f}, $\sigma=${1:.2f})".format(mu, sigma), col])

# %% OUTLIERS
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

ensemble = pd.concat([train, test], axis=0)

le = LabelEncoder()
for col in ensemble.select_dtypes(include='object').columns:
    ensemble[col] = le.fit_transform(ensemble[col])

train = ensemble.iloc[:train.shape[0], :]
test = ensemble.iloc[train.shape[0]:, :]

isol_forest = IsolationForest(max_features=train.shape[1])
isol_forest.fit(train)
idx_to_rm = list(isol_forest.predict(train))
indeces_to_rm = [index for index, val in enumerate(idx_to_rm) if val == -1]
indeces_to_rm
X_train.drop(index=indeces_to_rm, inplace=True)
y_train.drop(index=indeces_to_rm, inplace=True)

# test part
idx_to_rm = list(isol_forest.predict(test))
indeces_to_rm = [index for index, val in enumerate(idx_to_rm) if val == -1]
indeces_to_rm = np.array(indeces_to_rm) + train.shape[0]

X_test.drop(index=indeces_to_rm, inplace=True)
y_test.drop(index=indeces_to_rm, inplace=True)

# %% FEATURE SELECTION
train = pd.concat([X_train, y_train], axis=1)
correlations = train.corr()
n_highest_corrs = 10
columns = (correlations.nlargest(n_highest_corrs, 'SalePrice')['SalePrice']
           .index)
corr_mat = np.corrcoef(train[columns].values.T)
sns.set(font_scale=1)
plt.subplots(figsize=(10, 9))
hm = sns.heatmap(corr_mat, cbar=True, annot=True, fmt='.2f',
                 annot_kws={'size': 9},
                 cmap="coolwarm",
                 yticklabels=columns.values,
                 xticklabels=columns.values)
plt.title("Heatmap for 10 variables most correlated with SalePrice.",
          fontsize=15)
plt.show()

cols_to_drop = ['TotRmsAbvGrd',
                'GarageArea',
                '1stFlrSF',
                'YearRemodAdd',
                'SalePrice']

cols_continuous = [col for col in columns if col not in cols_to_drop]
X_train_continuous = X_train[cols_continuous]
X_test_continuous = X_test[cols_continuous]

ccc = CustomCategoricalChooser(n_features=45, response_var='SalePrice')
params = ccc.fit(train)
params
cols_to_keep = ['OverallQual', 'Neighborhood', 'Foundation']
X_train_categorical = X_train[cols_to_keep]
X_train_categorical
X_test_categorical = X_test[cols_to_keep]


# %% SAVE DATA
pd.concat([X_train_categorical,
           X_train_continuous,
           y_train],
          axis=1).to_csv(os.getcwd()+"/analysis_data/train.csv")
pd.concat([X_test_categorical,
           X_test_continuous,
           y_test],
          axis=1).to_csv(os.getcwd()+"/analysis_data/test.csv")
