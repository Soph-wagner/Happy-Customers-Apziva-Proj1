'''
Author: Sophia Wagner
Date: 6/24/2024
Description: Hyperparameter Tuning for Random Forest and SGD Classifier Models, using Hyperopt
'''

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Importing the dataset
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

# making sure data is loaded
print(data.head(5))

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# small X with only 3 features: 'X1', 'X3', 'X6'
X_small = data[['X1', 'X3', 'X6']].values

# creating Min-Max scaler
scaler = MinMaxScaler()
X_scale_minmax = scaler.fit_transform(X)

# defining objective function for Random Forest
def rf_objective(params):
    params = { 
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'max_features': str(params['max_features']),
        'criterion': str(params['criterion'])
    }

    rf_model = RandomForestClassifier(**params)
    rf_score = cross_val_score(rf_model, X, y, cv=10, scoring='accuracy').mean()
    return {'loss': -rf_score, 'status': STATUS_OK}

# defining objective function for SGD Classifier
def sgd_objective(params):
    params = { 
        'alpha': params['alpha'],
        'loss': str(params['loss']),
        'penalty': str(params['penalty']),
        'max_iter': int(params['max_iter'])
    }

    sgd_model = SGDClassifier(**params)
    sgd_score = cross_val_score(sgd_model, X, y, cv=5, scoring='accuracy').mean()
    return {'loss': -sgd_score, 'status': STATUS_OK}

# defining the hyperparameter space for both models
## starting with random forest search space
rf_space = { 
    'n_estimators': hp.quniform('n_estimators', 10, 1000, 10),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}

## now SGD Classifier search space
sgd_space = {
    'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(1)),
    'loss': hp.choice('loss', ['hinge', 'log_loss', 'squared_hinge', 'squared_error']),
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
    'max_iter': hp.quniform('max_iter', 400000, 500000, 10000)
}

# RUNNING the Optimization !!
rf_best = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=100, trials=Trials())
print("Finished Random Forest Optimization...")

sgd_best = fmin(fn=sgd_objective, space=sgd_space, algo=tpe.suggest, max_evals=100, trials=Trials())
print("Finished SGD Classifier Optimization...")

print("Best Random Forest Hyperparameters: ", rf_best)
print("Best Loss for Random Forest: ", rf_best['loss'])
print("Best SGD Classifier Hyperparameters: ", sgd_best)
print("Best Loss for SGD Classifier: ", sgd_best['loss'])

