#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:58:41 2023

@author: alina
"""

# -----------------------------------------------------------------------------
# -----------------------------Flights analysis: MaxVotingClassifier-----------
# -----------------------------------------------------------------------------


#%% 
# Import needed modules
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from model_definition import *
from sklearn.ensemble import VotingClassifier

import pickle

#%%
# Load models
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

best_rf_a = pickle.load(open('models/rand_forest_a.sav', 'rb'))
best_rf_b = pickle.load(open('models/rand_forest_b.sav', 'rb'))

best_xgb_a = pickle.load(open('models/xg_boost_a.sav', 'rb'))
best_xgb_b = pickle.load(open('models/xg_boost_b.sav', 'rb'))

best_cbc_a = pickle.load(open('models/cat_boost_a.sav', 'rb'))
best_cbc_b = pickle.load(open('models/cat_boost_b.sav', 'rb'))

best_lr_a = pickle.load(open('models/log_reg_a.sav', 'rb'))
best_lr_b = pickle.load(open('models/log_reg_b.sav', 'rb'))


#%%
# Use previous results for classifiers to see results for set "a" and "b"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


estimators_a = [('rf',best_rf_a), ('xgb',best_xgb_a), ('cbc',best_cbc_a), 
                ('lr',best_lr_a)
                ]

best_vc_a = VotingClassifier(
    estimators=estimators_a, 
    voting='soft', 
    weights=[1,1,1,1]
)

estimators_b = [('rf',best_rf_b), ('xgb',best_xgb_b), ('cbc',best_cbc_b), 
                ('lr',best_lr_b)
                ]

best_vc_b = VotingClassifier(
    estimators=estimators_b, 
    voting='soft', 
    weights=[1,1,1,1]
)

#%%

# Add vc to the dictionary 
model['best_vc_a'] = best_vc_a
model['best_vc_b'] = best_vc_b

#%%

modelfit('best_vc_a', x_train_a, y_train_a, x_test_a, y_test_a, **model)

#%%

modelfit('best_vc_b', x_train_b, y_train_b, x_test_b, y_test_b, **model)


