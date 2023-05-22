#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:16:39 2023

@author: alina
"""


# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: XGBClassifier-Experimental----
# -----------------------------------------------------------------------------


#%% 
# Import the module to train and optimize the model, and to print results
# -----------------------------------------------------------------------------


from model_definition import *
from sklearn.model_selection import train_test_split

#%%

# Load preprocessed experimental data and amend the type of features
#------------------------------------------------------------------------------


inputs_exp = pd.read_csv("src/inputs_exp.csv")
targets_exp = pd.read_csv("src/targets_exp.csv")
targets_exp = targets_exp["Class"]


targets_exp = targets_exp.astype("category")

cat_feat = inputs_exp.drop(["Length"], axis=1).astype("category") 
cat_features = np.where(inputs_exp.dtypes=='category')[0]


# Split data into train and test, and shuffle
#------------------------------------------------------------------------------


x_train_exp, x_test_exp, y_train_exp, y_test_exp = train_test_split( 
    inputs_exp, 
    targets_exp, 
    train_size=0.8, 
    random_state=20
)

#%%
# Model fit
# -----------------------------------------------------------------------------

xgb_exp = xgb_cl.set_params(**{'scale_pos_weight':1.09})
model['xgb_exp'] = xgb_exp

xgb_exp = modelfit('xgb_exp', x_train_exp, y_train_exp, x_test_exp, y_test_exp, **model)
xgb_exp
