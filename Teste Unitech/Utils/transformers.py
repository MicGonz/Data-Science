# transformers.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X)

    def get_feature_names_out(self, input_features=None):
        return [f"{f}" for f in input_features]
