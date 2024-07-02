'''
Author: Sophia Wagner
Date: 6/14/2024
Description: Random Forest Model for ACME Happiness Survey Data
'''

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing the dataset
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

# quick look at the data
print(data.head())

#splitting the data into features and target variable
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)

#our target variable is categorical, so we will use a classification model

#Creating Random Forest Model
RF_model = RandomForestClassifier(n_estimators=100, random_state=224)

#fitting the model
RF_model.fit(X_train, y_train)

#predicting the test set
y_pred = RF_model.predict(X_test)

#Evaluating model
print("Confusion Matrix for Random Forest: ")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

print("Interpretation of Confusion Matrix: ")
print(" True Positives (TP): ", cm[1, 1])
print(" True Negatives (TN): ", cm[0, 0])
print(" False Positives (FP): ", cm[0, 1])
print(" False Negatives (FN): ", cm[1, 0])

print("Classification Report for Random Forest: ")
print(classification_report(y_test, y_pred, labels=[0, 1]))

RF_accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy: ", RF_accuracy)

#now Looking at Feature Importance
feature_importance = pd.Series(RF_model.feature_importances_, index=data.columns[1:]).sort_values(ascending=False)
print("Feature Importance: ", feature_importance)