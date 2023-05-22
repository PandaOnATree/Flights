#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 00:12:44 2023

@author: alina
"""


# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: Module for scalers------------
# -----------------------------------------------------------------------------


#%% 
# Import modules
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler


#%%
# Standard scaler
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Create a class of a standard scaler (for data without outliers):
class CustomScalerStandard(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        
    def fit(self, x, y=None):
        self.scaler.fit(x[self.columns], y)
        return self
    
    def transform(self, x, y=None):
        init_col_order = x.columns
        x_scaled = pd.DataFrame(self.scaler.transform(x[self.columns]), 
                                columns=self.columns
                               )
        x_not_scaled = x.loc[:,~x.columns.isin(self.columns)]
        return pd.concat([x_not_scaled, x_scaled], axis=1)[init_col_order]
    
#%%
# Robust scaler
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Create a class of a robust scaler (for data with outliers):
class CustomScalerRobust(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.scaler = RobustScaler()
        self.columns = columns
        
    def fit(self, x, y=None):
        self.scaler.fit(x[self.columns], y)
        return self
    
    def transform(self, x, y=None):
        init_col_order = x.columns
        x_scaled = pd.DataFrame(self.scaler.transform(x[self.columns]), 
                                columns=self.columns
                               )
        x_not_scaled = x.loc[:,~x.columns.isin(self.columns)]
        return pd.concat([x_not_scaled, x_scaled], axis=1)[init_col_order]
    
    
    