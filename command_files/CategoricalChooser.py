from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import scipy.stats as stats
import math


class CustomCategoricalChooser(TransformerMixin, BaseEstimator):
    def __init__(self, n_features, response_var):
        self.n_features = n_features
        self.cols_to_keep = []
        self.response_var = response_var

    def transform(self, X):
        return X[self.cols_to_keep]

    def fit(self, X, *_):
        df = X.select_dtypes(include='object')
        params = []
        for col in df.columns:
            unique_vals = df[col].unique()
            test = []
            for col_n in unique_vals:
                test.append(X[self.response_var][df[col] == col_n])
            stat, p = stats.f_oneway(*test)
            if not math.isnan(p):
                params.append((col, p))
        params.sort(key=lambda x: x[1])
        return params[:self.n_features]
