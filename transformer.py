import pandas as pd
import sklearn 
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# Update Outliers
class UpdateOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_cols, limits):
        self.outlier_cols = outlier_cols
        self.limits = limits
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for i in range(len(self.outlier_cols)):
            X.loc[X[self.outlier_cols[i]] > self.limits[i], self.outlier_cols[i]] = self.limits[i]  
        return X
    
# Recalculate Computed Columns
class ReCalcNumCols(BaseEstimator, TransformerMixin):
    def __init__(self, main_columns, sub_columns, columns):
        self.main_columns = main_columns
        self.sub_columns = sub_columns
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for i in range(len(self.main_columns)):
            for r in X:
                x = self.columns.index(self.main_columns[i])
                x1, x2 = self.columns.index(self.sub_columns[i][0]), self.columns.index(self.sub_columns[i][1])
                r[x] = r[x1] / r[x2] * 100 if r[x2] > 0 else 0            
        return X
    
# Drop columns after recalculation
class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, del_cols, columns):
        self.columns = columns
        self.del_cols = del_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        del_idx = []
        for col in self.del_cols:
            del_idx.append(self.columns.index(col))
        X = np.delete(X, del_idx, 1)
        return X
