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
    - XGBClassifier
    - CatBoostClassifier
'''

# Import libraries for general purpose
#------------------------------------------------------------------------------


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

from datetime import datetime

import os.path

import joblib



# Load preprocessed data and amend the type of features
#------------------------------------------------------------------------------


inputs = pd.read_csv("inputs.csv")
targets = pd.read_csv("targets.csv")
targets = targets["Class"]


#targets = targets.astype("category")

#cat_feat = inputs.drop(["Length"], axis=1).astype("category") 
cat_features = np.where(inputs.dtypes=='category')[0]


# Split data into train and test, and shuffle
#------------------------------------------------------------------------------


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( 
    inputs, 
    targets, 
    train_size=0.8, 
    random_state=20
)

# print(x_train_a.shape, x_test_a.shape, y_train_a.shape, y_test_a.shape)


print("All preprocessed data, split into train and test loaded succesfully.")


# Load libraries to model training and hyperparameter tunning
#------------------------------------------------------------------------------



from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost import XGBClassifier

import catboost
from catboost import CatBoostClassifier
from catboost import cv, Pool

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

import functools

from sklearn.metrics import make_scorer, fbeta_score, brier_score_loss

#from sklearn.metrics import roc_curve, precision_recall_curve


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import seaborn as sns
plt.style.use('default')

from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

# Define all considered models
#------------------------------------------------------------------------------


xgb_cl = XGBClassifier( 
    booster='gbtree',
    learning_rate=0.2,
    n_estimators=300,
    max_depth=6,
    min_child_weight=0.8,
    enable_categorical=True,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1.09,
    seed=30,
    max_cat_to_onehot=1,
    tree_method="hist",
    reg_alpha=0.8,
    eval_metric=["error", "logloss"]
    #eval_metric='auc',
)

cbc_cl = CatBoostClassifier(
    class_weights=[1,1.09],
    iterations=300, 
    depth=7,
    loss_function='Logloss', 
    eval_metric='Logloss',
    random_seed=27, 
    subsample=0.9,
    logging_level='Silent'
)

rf_cl = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=27,
    max_features=10,
    class_weight={0:1, 1:1.09},
    n_jobs=-1,
    criterion='log_loss'
    )

# Define a dictionary for the models which will be constantly updated
model = {}

model['xgb'] = xgb_cl

model['cbc'] = cbc_cl

model['rf'] = rf_cl


print("All basic versions of considered models loaded succesfully.")


# Define functions needed for training and tunning
#------------------------------------------------------------------------------


# Define a function for basic model training and printing results
def modelfit(name_of_mod, x_train, y_train, x_test, y_test, 
             cat_features=cat_features, **model):
    
    if type(name_of_mod) is not str:
        print("name_of_mod has to be of a string type!")
        
    dir_name = 'models_results'
    
    start = datetime.now()
    
    alg = model[name_of_mod]
    
    # Fitting and predicting
    #**************************************************************************
    if 'xgb' in name_of_mod.split('_'):
    
        # Fit the algorithm on the data
        eval_set = [(x_train, y_train), (x_test, y_test)]
        alg.fit(x_train, y_train, eval_set=eval_set, verbose=False)
            
        # Predict training set:
        y_pred_train = alg.predict(x_train)
        y_predprob_train = alg.predict_proba(x_train)[:,1]
        
        #Evaluate test set:
        y_pred = alg.predict(x_test)
        y_predprob = alg.predict_proba(x_test)[:,1]
        
        results = alg.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        
        
        # Plot tests for overfitting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
        fig.suptitle('Tests for overfitting for {}'.format(name_of_mod), 
                     fontsize=16, y=1.1)
        
        # Plot log loss
        ax1.plot(x_axis, results['validation_0']['logloss'], label='Train',
                 color='green'
                 )
        ax1.plot(x_axis, results['validation_1']['logloss'], label='Test',
                 color='orange'
                 )
        ax1.legend()
        ax1.set_ylabel('Log Loss')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_xlabel('n_estimators')
        ax1.set_title('XGBoost Log Loss')
        
        # Plot classification max error
        ax2.plot(x_axis, results['validation_0']['error'], label='Train',
                 color='green'
                 )
        ax2.plot(x_axis, results['validation_1']['error'], label='Test',
                 color='orange'
                 )
        ax2.legend()
        ax2.set_ylabel('Classification Error')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.set_xlabel('n_estimators')
        ax2.set_title('XGBoost Classification Error')
        
        # Save the plots
        path_fig_test = dir_name+'/'+name_of_mod+'_test'+'.png'
        plt.savefig(path_fig_test, dpi=300, bbox_inches='tight')
        
        
    elif 'rf' in name_of_mod.split('_'):
        
        # Fit the algorithm on the data
        alg.fit(x_train, y_train)
        
        # Predict training set:
        y_pred_train = alg.predict(x_train)
        y_predprob_train = alg.predict_proba(x_train)[:,1]
        
        #Evaluate test set:
        y_pred = alg.predict(x_test)
        y_predprob = alg.predict_proba(x_test)[:,1]
        
        
    else:
        # Define pool for cbc
        train_pool = Pool(x_train, y_train, cat_features)
        test_pool = Pool(x_test, y_test, cat_features)
        
        # Fit the algorith on data
        alg.fit(train_pool)
        y_pred_train = alg.predict(train_pool)
        y_predprob_train = alg.predict_proba(train_pool)[:,1]
        
        #Evaluate test set:
        y_pred = alg.predict(test_pool)
        y_predprob = alg.predict_proba(test_pool)[:,1]
    
    
    #Calculate metrics:
    #**************************************************************************
    accuracy_train = metrics.accuracy_score(y_train.values, y_pred_train)
    accuracy_test = metrics.accuracy_score(y_test.values, y_pred)
    
    auc_score_train = metrics.roc_auc_score(y_train.values, y_predprob_train)
    auc_score_test = metrics.roc_auc_score(y_test.values, y_predprob)
    
    logloss_test = metrics.log_loss(y_test.values, y_predprob)
    
    rec_test = metrics.recall_score(y_test.values, y_pred)
    prec_test = metrics.precision_score(y_test.values, y_pred)
    f1_test = metrics.f1_score(y_test.values, y_pred)
    
    cohen_test = metrics.cohen_kappa_score(y_test.values, y_pred)
    mat_test = metrics.matthews_corrcoef(y_test.values, y_pred)
    
    
    # Create a file with the report on the metrics
    #**************************************************************************
    name = name_of_mod+" - model report:"
      
    path_txt = dir_name+'/'+name_of_mod+'.txt'
        
    file = open(path_txt, 'w+')
        
        
    file.write('***'*30+'\n')
    file.write(str(name)+'\n')
    file.write('---'*30+'\n')
    file.write('\n')
        
    file.write('Model: {}'.format(alg)+'\n')
    file.write('\n')
        
    file.write("\nRecall (Test): {:.4f}".format(rec_test)+'\n')
    file.write("Precision (Test): {:.4f}".format(prec_test)+'\n')
    file.write("F1 score (Test): {:.4f}".format(f1_test)+'\n')  
        
    file.write("\nCohen Kappa (Test): {:.4f}".format(cohen_test)+'\n')
    file.write("MCC (Test): {:.4f}".format(mat_test)+'\n')
        
    file.write("\nAUC Score (Train): {:.4f}".format(auc_score_train)+'\n')
    file.write("AUC Score (Test): {:.4f}".format(auc_score_test)+'\n')
        
    file.write("Accuracy (Train): {:.4f}".format(accuracy_train)+'\n')
    file.write("Accuracy (Test): {:.4f}".format(accuracy_test)+'\n')
    
    
    file.write("\nLogLoss (Test): {:.4f}".format(logloss_test)+'\n')
    

    file.write('\n')
        
        
    file.write('***'*30+'\n')
        
    file.close()

    

    #Print model report:
    #**************************************************************************
    print('***'*20)
    print(name)
    print('---'*20)
    
    print("\nRecall (Test): {:.4f}".format(rec_test))
    print("Precision (Test): {:.4f}".format(prec_test))
    print("F1 score (Test): {:.4f}".format(f1_test))
    
    print("\nCohen Kappa (Test): {:.4f}".format(cohen_test))
    print("MCC (Test): {:.4f}".format(mat_test))
    
    print("\nAUC Score (Train): {:.4f}".format(auc_score_train))
    print("AUC Score (Test): {:.4f}".format(auc_score_test))
    
    print("\nAccuracy (Train): {:.4f}".format(accuracy_train))
    print("Accuracy (Test): {:.4f}".format(accuracy_test))
      
    print("\nLogLoss (Test): {:.4f}".format(logloss_test))
      
    # Plots
    #**************************************************************************
        
    # For each single model plot feature importances and confusion matrix
    #**************************************************************************
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
    fig.suptitle('Results for {}'.format(name_of_mod), fontsize=16, y=1.1)
    
    # Create a series containing feature importances from the model 
    # and feature names from the training data
    feature_importances = pd.Series(alg.feature_importances_, 
                                    index=x_train.columns).sort_values(ascending=False)
    
    # Subplot a simple bar chart for feature_importances
    ax1.bar(feature_importances.index, feature_importances.values, 
            color='green'
            )
    ax1.set_title("Feature importances")
    ax1.tick_params(axis='x', rotation=90)
    

    # Create the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred) 
    cmp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    # Sublot confusion matrix
    cmp.plot(ax=ax2, colorbar=False, cmap='viridis')
    
    val_min = y_test.size * 0
    val_max = y_test.size * 0.55
    
    cmp.ax_.get_images()[0].set_clim(val_min, val_max)

    ax2.set_title("Confusion matrix")
        
    cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 
                        0.015, ax2.get_position().height]
                       )
    #img = plt.imshow(cmp, cmap='summer')
    plt.colorbar(cmp.im_, cax=cax)
    #.clim(vmin=200, vmax=500)


        
    # Save the plot in a separate directory
    path_fig_results = dir_name+'/'+name_of_mod+'_results'+'.png'
    plt.savefig(path_fig_results, dpi=300, bbox_inches='tight')
    
            
    # For each model plot precision vs recall and roc curve
    #************************************************************************** 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle('Precision-Recall and ROC curve generated by {}'
                 .format(name_of_mod), fontsize=16, y=1.1
                 )
        
    # Precision vs recall
    prec, recall, tresh = precision_recall_curve(y_test, y_pred, 
                                                 pos_label=alg.classes_[1]
                                                 )
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
    pr_display.plot(ax=ax1, color='green')

    # ROC curve
    fpr, tpr, tresholds = roc_curve(y_test, y_pred, 
                                    pos_label=alg.classes_[1]
                                    )
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot(ax=ax2, color='green')
        
    # Save the plots
    path_fig_pr_roc = dir_name+'/'+name_of_mod+'_pr_roc'+'.png'
    plt.savefig(path_fig_pr_roc, dpi=300, bbox_inches='tight')
    
    
    # For each model plot distribution of probability: real vs predicted
    #**************************************************************************
        
    fig, (ax1, ax2)  = plt.subplots(1,2, figsize=(10,3))
    fig.suptitle('Distribution of probability generated by {}'
                 .format(name_of_mod), fontsize=16, y=1.1
                 )
        
    # Plot distribution for 50 bins
    ax1.hist(y_test, bins=40, alpha=0.7, range=(-0.01, 1.01), 
             label="y_true", ec="black", fc="orange"
             )     
    ax1.hist(alg.predict_proba(x_test)[:,1], bins=40, alpha=0.7, 
             range=(-0.01, 1.01), label='y_pred', 
             ec="black", fc="green"
             )
        
    ax1.set_xlabel('Probability value (n of bins = 50)')
    ax1.set_ylabel('Probability density')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.legend(loc='best', bbox_to_anchor=(0.35, 0.45, 0.5, 0.5))
        
        
    # Plot distribution for 6 bins
    ax2.hist(y_test, bins=6, alpha=0.7, range=(0,1),
             label="y_true", ec="black", fc="orange"
             )
    ax2.hist(alg.predict_proba(x_test)[:,1], bins=6, alpha=0.7, 
             range=(0,1), label='y_pred', ec="black", fc="green"
             ) 
        
    ax2.set_xlabel('Probability value (n of bins = 6)')
    ax2.set_ylabel('Probability density')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.legend(loc='best', bbox_to_anchor=(0.3, 0.48, 0.5, 0.5))
        
    # Save the plots
    path_fig_distr = dir_name+'/'+name_of_mod+'_distr'+'.png'
    plt.savefig(path_fig_distr, dpi=300, bbox_inches='tight')
    
    
    #**************************************************************************
        
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
    
    if est == cbc_cl:
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

print("All functions needed for training and tunning hyperparameters loaded succesfully.")

# -----------------------------------------------------------------------------
