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


from model_definition import *

#%%
# Basic model fit
# -----------------------------------------------------------------------------

xgb_cl = modelfit('xgb', x_train, y_train, x_test, y_test, **model)
xgb_cl

#%%
# Hyperparameters tunning
# Use GridSearchCV to find the best hyperparameters
# -----------------------------------------------------------------------------

print("Tunning hyperparameters for xgb..........")

scorer='recall' 
#scorer=make_scorer(log_loss)

# Test 1: tunning 'max_depth' and 'min_child_weight'

param_test1 = {'max_depth':[5,6,7,8], 'min_child_weight':[0.8,1]}

gsearch_xgb = GridSearchCV(
    estimator=xgb_cl, 
    param_grid=param_test1, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1,
    refit=True
)

print("Tunning 'max_depth' and 'min_child_weight'..........")
best_xgb = search_fit(gsearch_xgb, x_train, y_train)
best_xgb

#%%

# Test 2: tunning 'subsample' and 'colsample_bytree'

param_test2 = {'subsample':[i/10.0 for i in range(7,10)], 
               'colsample_bytree':[i/10.0 for i in range(7,9)]
               }

gsearch_xgb = GridSearchCV(
    estimator=best_xgb, 
    param_grid=param_test2, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1,
    refit=True
)

print("Tunning 'subsample' and 'colsample_bytree'..........")
best_xgb = search_fit(gsearch_xgb, x_train, y_train)
best_xgb

#%%

# Test 3: tunning 'gamma'

param_test3 = {'gamma':[i/10.0 for i in range(0,2)]}

gsearch_xgb = GridSearchCV(
    estimator=best_xgb, 
    param_grid=param_test3, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1,
    refit=True
)

print("Tunning 'gamma'..........")
best_xgb = search_fit(gsearch_xgb, x_train, y_train)
best_xgb

#%%

# Test 4: tunning 'reg_alpha'

param_test4 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch_xgb = GridSearchCV(
    estimator=best_xgb, 
    param_grid=param_test4, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

print("Tunning 'reg_alpha'..........")
best_xgb = search_fit(gsearch_xgb, x_train, y_train)
best_xgb

#%%

# Test 5: tunning 'learning_rate' and 'n_estimators'

param_test5 = {'learning_rate':[0.05, 0.1, 0.15, 0,2, 0.3], 'n_estimators':[200, 300, 500, 1000]}

gsearch_xgb = GridSearchCV(
    estimator=best_xgb, 
    param_grid=param_test5, 
    scoring=scorer, 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

print("Tunning 'learning_rate' and 'n_estimators'..........")
best_xgb = search_fit(gsearch_xgb, x_train, y_train)
best_xgb

#%%

# Best model
# -----------------------------------------------------------------------------


# Add the best models to the dictionary
model['best_xgb'] = best_xgb

best_xgb = modelfit('best_xgb', x_train, y_train, x_test, y_test, **model)
best_xgb


#%%
# Save model
# -----------------------------------------------------------------------------


joblib.dump(best_xgb, 'models/xg_boost.joblib')




