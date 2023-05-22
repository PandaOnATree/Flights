#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:57:52 2023

@author: alina
"""

# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: RandomForestClassifier--------
# -----------------------------------------------------------------------------


#%% 
# Import the module to train and optimize the model, and to print results
# -----------------------------------------------------------------------------


from model_definition import *


#%%
# Basic model fit
# -----------------------------------------------------------------------------


rf_cl = modelfit('rf', x_train, y_train, x_test, y_test, **model)
rf_cl

#%%
# Hyperparameters tunning
# -----------------------------------------------------------------------------


print("Tunning hyperparameters for rf..........")

scorer='recall'

# Use GridSearchCV to find the best hyperparameters

param_rf_test1 = {'n_estimators':[100,200,300], 'max_depth':[5,6,7]}

gsearch_rf = GridSearchCV(
    estimator=rf_cl, 
    param_grid=param_rf_test1, 
    scoring=scorer,
    cv=5,
    n_jobs=-1, 
    refit=True
)

print("Tunning 'n_estimators' and 'max_depth'..........")
best_rf = search_fit(gsearch_rf, x_train, y_train)

#%%

# Best model
# -----------------------------------------------------------------------------


model['best_rf'] = best_rf

best_rf = modelfit('best_rf', x_train, y_train, x_test, y_test, **model)

#%%
# Save models
# -----------------------------------------------------------------------------


joblib.dump(best_rf, 'models/rand_forest.joblib')


