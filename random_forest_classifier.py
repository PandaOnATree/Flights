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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from model_definition import *

#%%
# Basic model fit for set "a" and "b"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#%%

rf_a = modelfit('rf', x_train_a, y_train_a, x_test_a, y_test_a, **model)
rf_a

#%%

rf_b = modelfit('rf', x_train_b, y_train_b, x_test_b, y_test_b, **model)
rf_b 


#%%
# Hyperparameters tunning
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("Tunning hyperparameters for rf..........")

# Use RandomizedSearchCV to find the best hyperparameters

param_dist = {'n_estimators':randint(50,500), 'max_depth':randint(1,20)}

rsearch_rf_a = RandomizedSearchCV(
    estimator=rf_a, 
    param_distributions=param_dist, 
    n_iter=5, 
    scoring='roc_auc',
    cv=5, 
    refit=True
)

rsearch_rf_b = RandomizedSearchCV(
    estimator=rf_b, 
    param_distributions=param_dist, 
    n_iter=5, 
    scoring='roc_auc', 
    cv=5, 
    refit=True
)
#%%

print("Tunning 'n_estimators' and 'max_depth' for dataset 'a'..........")
best_rf_a = search_fit(rsearch_rf_a, x_train_a, y_train_a)

#%%

print("Tunning 'n_estimators' and 'max_depth' for dataset 'b'..........")
best_rf_b = search_fit(rsearch_rf_b, x_train_b, y_train_b)

#%%

# Best model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


model['best_rf_a'] = best_rf_a
model['best_rf_b'] = best_rf_b

#%%

best_rf_a = modelfit('best_rf_a', x_train_a, y_train_a, x_test_a, y_test_a, **model)

#%%

best_rf_b = modelfit('best_rf_b', x_train_b, y_train_b, x_test_b, y_test_b, **model)

#%%
# Save models
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

rand_forest_a = 'models/rand_forest_a.sav'
pickle.dump(best_rf_a, open(rand_forest_a, 'wb'))

rand_forest_b = 'models/rand_forest_b.sav'
pickle.dump(best_rf_b, open(rand_forest_b, 'wb')) 

