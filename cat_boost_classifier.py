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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from model_definition import *

#%%
# Basic model fit for set "a" and "b"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#%%

cbc_a = modelfit('cbc', x_train_a, y_train_a, x_test_a, y_test_a, **model)
cbc_a

#%%

cbc_b = modelfit('cbc', x_train_b, y_train_b, x_test_b, y_test_b, **model)
cbc_b

#%%
# Hyperparameters tunning
# Use GridSearchCV to find the best hyperparameters
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("Tunning hyperparameters for cbc..........")

# Test 1: tunning 'depth' and 'learning_rate'

param_cbc_test1 = {'depth':[9,10], 'learning_rate':[0.08,0.1]}

gsearch_cbc = GridSearchCV(
    estimator=cbc, 
    param_grid=param_cbc_test1, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

#%%

print("Tunning 'depth' and 'learning_rate' for dataset 'a'..........")
best_cbc_a = search_fit(gsearch_cbc, x_train_a, y_train_a)
best_cbc_a

#%%

print("Tunning 'depth' and 'learning_rate' for dataset 'b'..........")
best_cbc_b = search_fit(gsearch_cbc, x_train_b, y_train_b) 
best_cbc_b

#%%

# Test 2: tunning 'subsample' and 'iterations'

param_cbc_test2 = {'subsample':[0.8,0.9], 'iterations':[100,2000]}

gsearch_cbc_a = GridSearchCV(
    estimator=best_cbc_a, 
    param_grid=param_cbc_test2, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

gsearch_cbc_b = GridSearchCV(
    estimator=best_cbc_b, 
    param_grid=param_cbc_test2, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1, 
    refit=True
)

#%%

print("Tunning 'subsample' and 'iterations' for dataset 'a'..........")
best_cbc_a = search_fit(gsearch_cbc_a, x_train_a, y_train_a) 
best_cbc_a

#%%

print("Tunning 'subsample' and 'iterations' for dataset 'b'..........")
best_cbc_b = search_fit(gsearch_cbc_b, x_train_b, y_train_b)
best_cbc_b

#%%

# Best model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Add the best models to the dictionary
model['best_cbc_a'] = best_cbc_a
model['best_cbc_b'] = best_cbc_b

#%%

best_cbc_a = modelfit('best_cbc_a', x_train_a, y_train_a, x_test_a, y_test_a, **model)
best_cbc_a

#%%

best_cbc_b = modelfit('best_cbc_b', x_train_b, y_train_b, x_test_b, y_test_b, **model)
best_cbc_b

#%%

# Best model with cross-validation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

'''
Consider, additionally, the best model of cbc, where the whole dataset (without splitting it) 
is taken into account and cross-validation is imposed on it
'''
from ipywidgets import IntSlider
display(IntSlider())

# Define a function to cross-validate the best cbc model

@time_passed
def modelfit_cbc_cv(alg, inputs, targets, cat_features, useTrainCV=True, 
                    cv_folds=5, early_stopping_rounds=50, plot=True):
    
    if useTrainCV:
        cbc_param = alg.get_params()
        train_pool = catboost.Pool(data=inputs, label=targets, 
                                   cat_features=cat_features, has_header=True
                                   )
        cvresult = cv(
            params=cbc_param, 
            pool=train_pool,
            fold_count=cv_folds,
            shuffle=True,
            partition_random_seed=0,
            plot=True,
            type='Classical',
            stratified=False
            #verbose=False
        )
        
    return cvresult
        
#%%

# Define cathegorical features for sets 'a' and 'b'
        
cat_features_a = np.where(inputs_a.dtypes==np.int64)[0]
cat_features_a

cat_features_b = np.where(inputs_b.dtypes==np.int64)[0]
cat_features_b

#%%

print("Fitting best cbc with cv for dataset 'a'..........")
modelfit_cbc_cv(best_cbc_a, inputs_a, targets_a, cat_features_a)


#%%

print("Fitting best cbc with cv for dataset 'b'..........")
modelfit_cbc_cv(best_cbc_b, inputs_b, targets_b, cat_features_b)

#%%

# Save models
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

cat_boost_a = 'models/cat_boost_a.sav'
pickle.dump(best_cbc_a, open(cat_boost_a, 'wb'))

cat_boost_b = 'models/cat_boost_b.sav'
pickle.dump(best_cbc_b, open(cat_boost_b, 'wb'))



