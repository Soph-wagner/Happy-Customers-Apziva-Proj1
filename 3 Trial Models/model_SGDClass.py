'''
Author: Sophia Wagner
Date: 6/16/2024
Description: Stochastic Gradient Descent Model for ACME Happiness Survey Data
'''

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importing the dataset
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

# making sure data is loaded
print(data.head(5))

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# creating Min-Max scaler
scaler = MinMaxScaler()
X_scale_minmax = scaler.fit_transform(X)

mm_X_train, mm_X_test, mm_y_train, mm_y_test = train_test_split(X_scale_minmax, y, test_size = 0.2, random_state=0)

# creating standardized scaler
scaler = StandardScaler()
X_scale_std = scaler.fit_transform(X)

std_X_train, std_X_test, std_y_train, std_y_test = train_test_split(X_scale_std, y, test_size = 0.2, random_state=0)

# creating SGD model
SGD_model = SGDClassifier(loss='hinge')
mm_SGD_model = SGDClassifier(loss='hinge')
std_SGD_model = SGDClassifier(loss='hinge')

# fitting the model
SGD_model.fit(X_train, y_train)

mm_SGD_model.fit(mm_X_train, mm_y_train)
std_SGD_model.fit(std_X_train, std_y_train)

# predicting the test set
y_pred = SGD_model.predict(X_test)

mm_y_pred = mm_SGD_model.predict(mm_X_test)
std_y_pred = std_SGD_model.predict(std_X_test)

# Evaluating model
print("Confusion Matrix for Regular SGD: ")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

print("Interpretation of Regular SGD Confusion Matrix: ")
print(" True Positives (TP): ", cm[1, 1])
print(" True Negatives (TN): ", cm[0, 0])
print(" False Positives (FP): ", cm[0, 1])
print(" False Negatives (FN): ", cm[1, 0])

print("Classification Report for Regular SGD: ")
print(classification_report(y_test, y_pred, labels=[0, 1]))

SGD_accuracy = accuracy_score(y_test, y_pred)
print("Reglar SGD Classifier Accuracy: ", SGD_accuracy)

# now Looking at Feature Importance
feature_importance = pd.Series(SGD_model.coef_[0], index=data.columns[1:]).sort_values(ascending=False)
print("Feature Importance: ", feature_importance)

# Eval for minmax model
print("Confusion Matrix for MinMax SGD: ")
cm = confusion_matrix(mm_y_test, mm_y_pred, labels=[0, 1])
print(cm)

print("Interpretation of MinMax SGD Confusion Matrix: ")
print(" True Positives (TP): ", cm[1, 1])
print(" True Negatives (TN): ", cm[0, 0])
print(" False Positives (FP): ", cm[0, 1])
print(" False Negatives (FN): ", cm[1, 0])

print("Classification Report for MinMax SGD: ")
print(classification_report(mm_y_test, mm_y_pred, labels=[0, 1]))

mm_SGD_accuracy = accuracy_score(mm_y_test, mm_y_pred)
print("MinMax SGD Classifier Accuracy: ", mm_SGD_accuracy)

# minmax model Feature importance
mm_feature_importance = pd.Series(mm_SGD_model.coef_[0], index=data.columns[1:]).sort_values(ascending=False)
print("MinMax Feature Importance: ", mm_feature_importance)

# Eval for standardized model
print("Confusion Matrix for Standardized SGD: ")
cm = confusion_matrix(std_y_test, std_y_pred, labels=[0, 1])
print(cm)

print("Interpretation of Standardized SGD Confusion Matrix: ")
print(" True Positives (TP): ", cm[1, 1])
print(" True Negatives (TN): ", cm[0, 0])
print(" False Positives (FP): ", cm[0, 1])
print(" False Negatives (FN): ", cm[1, 0])

print("Classification Report for Standardized SGD: ")
print(classification_report(std_y_test, std_y_pred, labels=[0, 1]))

std_SGD_accuracy = accuracy_score(std_y_test, std_y_pred)
print("Standardized SGD Classifier Accuracy: ", std_SGD_accuracy)

# standardized model Feature importance
std_feature_importance = pd.Series(std_SGD_model.coef_[0], index=data.columns[1:]).sort_values(ascending=False)
print("Standardized Feature Importance: ", std_feature_importance)