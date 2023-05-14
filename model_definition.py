#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:06:51 2023

@author: alina
"""
#------------------------------------------------------------------------------
#-----------------------------Flights analysis:Model creation------------------
#------------------------------------------------------------------------------

'''
In this file we define functions needed to perform data analysis and training 
for the following models: 
    - RandomForestClassifier
    - XGBClassifier
    - CatBoostClassifier
    - LogisticRegression
'''

# Import libraries for general purpose
#------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

from datetime import datetime

import os.path



# Load preprocessed data for sets "a" and "b"
#------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

inputs_a = pd.read_csv("inputs_a.csv")
targets_a = pd.read_csv("targets_a.csv")
targets_a = targets_a["Class"]

inputs_b = pd.read_csv("inputs_b.csv")
targets_b = pd.read_csv("targets_b.csv")
targets_b = targets_b["Class"]


# Split data into train and test, and shuffle
#------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(
    inputs_a, 
    targets_a, 
    train_size=0.8, 
    random_state=20
)

# print(x_train_a.shape, x_test_a.shape, y_train_a.shape, y_test_a.shape)

x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(
    inputs_b, 
    targets_b, 
    train_size=0.8, 
    random_state=20
)

# print(x_train_b.shape, x_test_b.shape, y_train_b.shape, y_test_b.shape)

print("All preprocessed data, split into trian and test loaded succesfully.")


# Load libraries to model training and hyperparameter tunning
#------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from sklearn import metrics

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

import catboost
from catboost import CatBoostClassifier
from catboost import cv


import xgboost as xgb
from xgboost import XGBClassifier

import functools

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define all considered models
#------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=27,
    max_features=1,
    n_jobs=-1
)

xgb = XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

cbc = CatBoostClassifier(
    iterations=100, 
    depth=5,
    loss_function='Logloss', 
    eval_metric='AUC', 
    random_seed=27, 
    subsample=0.8,
    logging_level='Silent'
)

lr = LogisticRegression(
    max_iter=100,
    solver='saga',
    random_state=27,
    n_jobs=-1
)

# Define a dictionary for the models which will be constantly updated
model = {}

model['rf'] = rf

model['xgb'] = xgb

model['cbc'] = cbc

model['lr'] = lr

print("All basic versions of considered models loaded succesfully.")


# Define functions needed for training and tunning
#------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Define a function for basic model training and printing results
def modelfit(name_of_mod, x_train, y_train, x_test, y_test, **model):
    
    if type(name_of_mod) is not str:
        print("name_of_mod has to be of a string type!")
    
    start = datetime.now()
    
    alg = model[name_of_mod]
    
    #Fit the algorithm on the data
    alg.fit(x_train, y_train)
        
    #Predict training set:
    y_pred_train = alg.predict(x_train)
    y_predprob_train = alg.predict_proba(x_train)[:,1]
    
    #Evaluate test set:
    y_pred = alg.predict(x_test)
    y_predprob = alg.predict_proba(x_test)[:,1]
    
    #Calculate metrics:
    accuracy_train = metrics.accuracy_score(y_train.values, y_pred_train)
    accuracy_test = metrics.accuracy_score(y_test.values, y_pred)
    
    auc_score_train = metrics.roc_auc_score(y_train.values, y_predprob_train)
    auc_score_test = metrics.roc_auc_score(y_test.values, y_predprob)
    
    logloss_test = metrics.log_loss(y_test.values, y_predprob)
    
    rec_test = metrics.recall_score(y_test.values, y_pred)
    prec_test = metrics.precision_score(y_test.values, y_pred)
    f1_test = metrics.f1_score(y_test.values, y_pred)
    
    
    # Create indices corresponding to the dataset to attach them when displaying reports
    if len(x_train.columns)==7:
        ind = '_a'
    else:
        ind = '_b'
    
    # For the best version of each model and set of data create a file with the report on the corresponding metrics
    list_name = name_of_mod.split('_')
    first_elem = list_name[0]
    
    dir_name = 'best_models_all'
    
    if first_elem=='best':
        name = name_of_mod+" - model report:"
        
        path_txt = dir_name+'/'+name_of_mod+'.txt'
        
        file = open(path_txt, 'w+')
        
        
        file.write('***'*30+'\n')
        file.write(str(name)+'\n')
        file.write('---'*30+'\n')
        file.write('\n')
        
        file.write('Model: {}'.format(alg)+'\n')
        file.write('\n')
        
        file.write("Accuracy (Train): {:.4f}".format(accuracy_train)+'\n')
        file.write("Accuracy (Test): {:.4f}".format(accuracy_test)+'\n')
    
        file.write("\nAUC Score (Train): {:.4f}".format(auc_score_train)+'\n')
        file.write("AUC Score (Test): {:.4f}".format(auc_score_test)+'\n')
    
        file.write("\nLogLoss (Test): {:.4f}".format(logloss_test)+'\n')
    
        file.write("\nRecall (Test): {:.4f}".format(rec_test)+'\n')
        file.write("Precision (Test): {:.4f}".format(prec_test)+'\n')
        file.write("F1 score (Test): {:.4f}".format(f1_test)+'\n')  
        file.write('\n')
        
        
        file.write('***'*30+'\n')

        
        file.close()
    else:
        name = name_of_mod+ind+" - model report:"
    

    #Print model report:
    print('***'*20)
    print(name)
    print('---'*20)
    print("\nAccuracy (Train): {:.4f}".format(accuracy_train))
    print("Accuracy (Test): {:.4f}".format(accuracy_test))
    
    print("\nAUC Score (Train): {:.4f}".format(auc_score_train))
    print("AUC Score (Test): {:.4f}".format(auc_score_test))
    
    print("\nLogLoss (Test): {:.4f}".format(logloss_test))
    
    print("\nRecall (Test): {:.4f}".format(rec_test))
    print("Precision (Test): {:.4f}".format(prec_test))
    print("F1 score (Test): {:.4f}".format(f1_test))

    
    if (name_of_mod=='best_vc_a' or name_of_mod=='best_vc_b'):
        print(' ')
    
    # For the LinearRegression create a table with coefficients and confusion matrix
    elif name_of_mod in ('lr', 'lr_a', 'lr_b', 'best_lr_a', 'best_lr_b'):
        
        intr = alg.intercept_
        coefs = alg.coef_
        feature_names = x_train.columns.values
        
        
        summary_table = pd.DataFrame(columns=["Feature Name"], data=feature_names)
        summary_table["Coefficients"] = np.transpose(coefs)

        # Adding an intercept in the first row:
        summary_table.index = summary_table.index+1
        summary_table.loc[0] = ["Intercept", intr[0]]
    
        summary_table = summary_table.sort_values("Coefficients", ascending=False)
        summary_table["Odds Ratio"] = np.exp(summary_table["Coefficients"])
        summary_table = summary_table.sort_values("Odds Ratio", ascending=False)
        
        print('---'*20)
        print(summary_table)
        print('---'*20)
        
        threshold=1.0

        # Subplot odds ratio
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
        fig.suptitle('Results for {}'.format(name_of_mod if "best" in name_of_mod.split("_") else name_of_mod+ind), fontsize=16, y=1.1)
        
        ax1.bar(summary_table["Feature Name"], summary_table["Odds Ratio"])
        ax1.tick_params(axis='x', rotation=90)
        ax1.title.set_text("Odds ratio for features")

        # Add the horizontal line indicating the threshold
        ax1.plot([-0.5,7.5],[threshold, threshold], "r-", label="threshold")
        ax1.legend(loc="upper right")
        
        # Sublot confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred) 
        cmp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        
        cmp.plot(ax=ax2, colorbar=False)
        ax2.set_title("Confusion matrix")
        
        cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 
                            0.015, ax2.get_position().height]
                           )
        plt.colorbar(cmp.im_, cax=cax)
        
        print("All features whose odds ratio is close to the threshold line have negligible impact on the target.")
        print("(This is equivalent to the corresponding coefficients being close to 0.)")
        print("These features should be, in principle, removed before building the model.")
        
        fig_name = name_of_mod+'.png'
        path_fig = dir_name+'/'+fig_name
        
        path = dir_name+'/'+name_of_mod+'.txt'
        
        # Save the figure for the "best" model.
        if (os.path.isfile(path)==True):
            plt.savefig(path_fig, dpi=250, bbox_inches='tight')
    
    # For other models (apart from LineaRegression) show feature importances and confusion matrix
    else: 
        # Create a series containing feature importances from the model and feature names from the training data
        feature_importances = pd.Series(alg.feature_importances_, 
                                        index=x_train.columns).sort_values(ascending=False)
        
        # Create the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred) 
        cmp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

        
        # Subplot a simple bar chart for feature_importances
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
        fig.suptitle('Results for {}'.format(name_of_mod if "best" in name_of_mod.split("_") else name_of_mod+ind), fontsize=16, y=1.1)

        
        ax1.bar(feature_importances.index, feature_importances.values)
        ax1.set_title("Feature importances")
        ax1.tick_params(axis='x', rotation=90)

        
        # Sublot confusion matrix
        cmp.plot(ax=ax2, colorbar=False)
        ax2.set_title("Confusion matrix")
        
        cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 
                            0.015, ax2.get_position().height]
                           )
        plt.colorbar(cmp.im_, cax=cax)
        
        
        # Save the plot in a separate directory
        fig_name = name_of_mod+'.png'
        path_fig = dir_name+'/'+fig_name
        
        path = dir_name+'/'+name_of_mod+'.txt'
        
        if (os.path.isfile(path)==True):
            plt.savefig(path_fig, dpi=250, bbox_inches='tight')

        
    stop = datetime.now()
    tot_time = stop-start
    print("\nTime passed: {}".format(tot_time))
    
    print('***'*20)
        
    return alg



# Define a decorator to display the time of execution of a function
def time_passed(input_function):
    
    @functools.wraps(input_function)
    
    def runtime_wrapper(*args, **kwargs):
        
        start = datetime.now() 
        return_value = input_function(*args, **kwargs)
        stop = datetime.now()
        time_passed = stop - start 
        
        print("\nFinished executing {} in {}.".format(input_function.__name__, 
time_passed))
        print("---"*20)
        
        return return_value
    
    return runtime_wrapper



# Define a fit function used for optimization
@time_passed
def search_fit(est, x_train, y_train):
    
    if est == cbc:
        train_pool = est.__init__pool(data=x_train, label=y_train, 
                                      has_header=True
                                      )
        est.fit(train_pool)
    
        print("Best parameters: {}".format(est.best_params_))
        print("Best score: {}".format(est.best_score_))
        
    else:
        est.fit(x_train, y_train)
    
        print("Best parameters: {}".format(est.best_params_))
        print("Best score: {}".format(est.best_score_))
    
    return est.best_estimator_


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


print("All functions needed for training and tunning hyperparameters loaded succesfully.")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
