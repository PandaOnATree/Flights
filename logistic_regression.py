#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:41:21 2023

@author: alina
"""

# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: LogisticRegression------------
# -----------------------------------------------------------------------------


#%% 
# Import the module to train and optimize the model, and to print results
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from model_definition import *
import pickle

#%%
# Basic model fit for set "a" and "b"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#%%

lr_a = modelfit('lr', x_train_a, y_train_a, x_test_a, y_test_a, **model)
lr_a

#%%

lr_b = modelfit('lr', x_train_b, y_train_b, x_test_b, y_test_b, **model)
lr_b

#%%
# Hyperparameters tunning
# Use GridSearchCV to find the best hyperparameters
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("Tunning hyperparameters for lr..........")

# Tunning 'C', 'penalty', and 'max_iter'

param_lr_test1 = {'C':np.linspace(0.01, 2., 4), 'penalty':['l1','l2'], 'max_iter':[100,200]}

gsearch_lr = GridSearchCV(
    estimator=lr, 
    param_grid=param_lr_test1, 
    scoring='roc_auc',
    n_jobs=-1, 
    cv=5, 
    refit=True)

#%%

print("Tunning 'C', 'penalty', and 'max_iter' for dataset 'a'..........")
best_lr_a = search_fit(gsearch_lr, x_train_a, y_train_a) 
best_lr_a

#%%

print("Tunning 'C', 'penalty', and 'max_iter' for dataset 'b'..........")
best_lr_b = search_fit(gsearch_lr, x_train_b, y_train_b) 
best_lr_b

#%%

# Best model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Add the best models to the dictionary
model['best_lr_a'] = best_lr_a
model['best_lr_b'] = best_lr_b

#%%

best_lr_a = modelfit('best_lr_a', x_train_a, y_train_a, x_test_a, y_test_a, **model)
best_lr_a

#%%

best_lr_b = modelfit('best_lr_b', x_train_b, y_train_b, x_test_b, y_test_b, **model)
best_lr_b

#%%

# Save models
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

log_reg_a = 'models/log_reg_a.sav'
pickle.dump(best_lr_a, open(log_reg_a, 'wb'))

log_reg_b = 'models/log_reg_b.sav'
pickle.dump(best_lr_b, open(log_reg_b, 'wb'))







