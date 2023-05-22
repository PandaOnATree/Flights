# Flights Analysis

## Short description 

In this project the chances of a flight being delayed or not are analysed. It is a binary classification problem. We train 3 models: XGBoostClassifier, CatBoostClassifier and RandomForestClassifier to predict whether a flights within the US is delayed or not. The dataset contains many categorical features, such as names of airports and names of airlines.

## Content

The project consists of the following parts. The exploratory analysis and data cleaning is included in {\it preprocessing_flights.ipynb}. The source files both before the preprocessing and generated after the preprocessing are included in the folder {\it src}. The main part for models training is given in {\it model_definition.py}, where the preprocessed data are split and shuffled, all functions for model's training and metric displaying are defined, and algorithms are initialized. The files: {\it xgb_classifier.py}, {\it cat_boost_classifier.py} and {\it random_forest_classifier.py} contain model training set-up and hyperparameters tunning. All three models are saved in the folder {\it models} and all results are kept in the folder {\it models_results}. The file {\it xgb_classifier_exp.py} icludes an experimental XGBoost model performing on data without instances cleaning to show how much uncleaned, noisy data affect the model performance.

## Results and models performance

The reports on models performances and various plots presenting 
