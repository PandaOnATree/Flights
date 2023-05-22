#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:43:20 2023

@author: alina
"""

# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: CatBoostClassifier------------
# -----------------------------------------------------------------------------


#%% 
# Import the module to train and optimize the model, and to print results
# -----------------------------------------------------------------------------


from model_definition import *


#%%
# Basic model fit
# -----------------------------------------------------------------------------


cbc_cl = modelfit('cbc', x_train, y_train, x_test, y_test, **model)
cbc_cl

#%%
# Hyperparameters tunning
# Use GridSearchCV to find the best hyperparameters
# -----------------------------------------------------------------------------


print("Tunning hyperparameters for cbc..........")

scorer='recall'

# Test 1: tunning 'depth' and 'learning_rate'

param_cbc_test1 = {'depth':[5,6,7,8], 'learning_rate':[0.08,0.1,0.2]}

gsearch_cbc = GridSearchCV(
    estimator=cbc_cl, 
    param_grid=param_cbc_test1, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

print("Tunning 'depth' and 'learning_rate'..........")
best_cbc = search_fit(gsearch_cbc, x_train, y_train)
best_cbc

#%%

# Test 2: tunning 'subsample' and 'iterations'

param_cbc_test2 = {'subsample':[0.8,0.9,1], 'iterations':[100,500,1000]}

gsearch_cbc = GridSearchCV(
    estimator=best_cbc, 
    param_grid=param_cbc_test2, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

print("Tunning 'subsample' and 'iterations'..........")
best_cbc = search_fit(gsearch_cbc, x_train, y_train) 
best_cbc

#%%

# Best model
# -----------------------------------------------------------------------------


# Add the best models to the dictionary
model['best_cbc'] = best_cbc

best_cbc = modelfit('best_cbc', x_train, y_train, x_test, y_test, **model)
best_cbc


#%%

# Save model
# -----------------------------------------------------------------------------


joblib.dump(best_cbc, 'models/cat_boost.joblib')



