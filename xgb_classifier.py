#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:02:40 2023

@author: alina
"""

# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: XGBClassifier-----------------
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

xgb_a = modelfit('xgb', x_train_a, y_train_a, x_test_a, y_test_a, **model)
xgb_a

#%%

xgb_b = modelfit('xgb', x_train_b, y_train_b, x_test_b, y_test_b, **model)
xgb_b

#%%
# Hyperparameters tunning
# Use GridSearchCV to find the best hyperparameters
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("Tunning hyperparameters for xgb..........")

# Test 1: tunning 'max_depth' and 'min_child_weight'

param_test1 = {'max_depth':[8,9,10], 'min_child_weight':[1,2]}

gsearch_xgb = GridSearchCV(
    estimator=xgb, 
    param_grid=param_test1, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

#%%

print("Tunning 'max_depth' and 'min_child_weight' for dataset 'a'..........")
best_xgb_a = search_fit(gsearch_xgb, x_train_a, y_train_a)
best_xgb_a

#%%

print("Tunning 'max_depth' and 'min_child_weight' for dataset 'b'..........")
best_xgb_b = search_fit(gsearch_xgb, x_train_b, y_train_b)
best_xgb_b

#%%

# Test 2: tunning 'subsample' and 'colsample_bytree'

param_test2 = {'subsample':[i/10.0 for i in range(8,10)], 
               'colsample_bytree':[i/10.0 for i in range(7,9)]
               }

gsearch_xgb_a = GridSearchCV(
    estimator=best_xgb_a, 
    param_grid=param_test2, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1,
    refit=True
)

gsearch_xgb_b = GridSearchCV(
    estimator=best_xgb_b, 
    param_grid=param_test2, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1,
    refit=True
)

#%%

print("Tunning 'subsample' and 'colsample_bytree' for dataset 'a'..........")
best_xgb_a = search_fit(gsearch_xgb_a, x_train_a, y_train_a)
best_xgb_a

#%%

print("Tunning 'subsample' and 'colsample_bytree' for dataset 'b'..........")
best_xgb_b = search_fit(gsearch_xgb_b, x_train_b, y_train_b)
best_xgb_b

#%%

# Test 3: tunning 'gamma'

param_test3 = {'gamma':[i/10.0 for i in range(0,3)]}

gsearch_xgb_a = GridSearchCV(
    estimator=best_xgb_a, 
    param_grid=param_test3, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1,
    refit=True
)

gsearch_xgb_b = GridSearchCV(
    estimator=best_xgb_b, 
    param_grid=param_test3, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1,
    refit=True
)

#%%

print("Tunning 'gamma' for dataset 'a'..........")
best_xgb_a = search_fit(gsearch_xgb_a, x_train_a, y_train_a)
best_xgb_a

#%%

print("Tunning 'gamma' for dataset 'b'..........")
best_xgb_b = search_fit(gsearch_xgb_b, x_train_b, y_train_b)
best_xgb_b

#%%

# Test 4: tunning 'reg_alpha'

param_test4 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch_xgb_a = GridSearchCV(
    estimator=best_xgb_a, 
    param_grid=param_test4, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

gsearch_xgb_b = GridSearchCV(
    estimator=best_xgb_b, 
    param_grid=param_test4, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

#%%

print("Tunning 'reg_alpha' for dataset 'a'..........")
best_xgb_a = search_fit(gsearch_xgb_a, x_train_a, y_train_a)
best_xgb_a

#%%

print("Tunning 'reg_alpha' for dataset 'b'..........")
best_xgb_b = search_fit(gsearch_xgb_b, x_train_b, y_train_b)
best_xgb_b

#%%

# Final tunning

'''
In the final version of the model we implement all best parameters found in tests 1-4. 
Additionally, we reduce the 'learning_rate' and increase 'n_estimators'.
'''

#%%

best_xgb_a = best_xgb_a.set_params(n_estimators=3000, learning_rate=0.02)

#%%

best_xgb_b = best_xgb_b.set_params(n_estimators=3000, learning_rate=0.02)

#%%

# Best model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Add the best models to the dictionary
model['best_xgb_a'] = best_xgb_a
model['best_xgb_b'] = best_xgb_b

#%%

best_xgb_a = modelfit('best_xgb_a', x_train_a, y_train_a, x_test_a, y_test_a, **model)
best_xgb_a

#%%

best_xgb_b = modelfit('best_xgb_b', x_train_b, y_train_b, x_test_b, y_test_b, **model)
best_xgb_b

#%%

# Best model with cross-validation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

'''
Consider, additionally, the best model of xgb, where the whole dataset (without splitting it) 
is taken into account and cross-validation is imposed on it
'''

# Define a function to cross-validate the best xgb model

@time_passed
def modelfit_xgb_cv(alg, inputs, targets, cv_folds=5, early_stopping_rounds=50):
    
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(data=inputs, label=targets)
    cvresult = xgb.cv(
        xgb_param, 
        xgtrain, 
        num_boost_round=10, 
        nfold=cv_folds,
        metrics='auc', 
        seed=100,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
    )

# Import xgboost once again to correctly perform modelfit with cv
import xgboost as xgb

#%%

modelfit_xgb_cv(model['best_xgb_a'], inputs_a, targets_a)

#%%

modelfit_xgb_cv(model['best_xgb_b'], inputs_b, targets_b)


#%%
# Save models
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

xg_boost_a = 'models/xg_boost_a.sav'
pickle.dump(best_xgb_a, open(xg_boost_a, 'wb'))

xg_boost_b = 'models/xg_boost_b.sav'
pickle.dump(best_xgb_b, open(xg_boost_b, 'wb'))




