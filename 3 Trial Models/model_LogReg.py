'''
Author: Sophia Wagner
Date: 6/14/2024
Description: Logistic Regression Model for ACME Happiness Survey Data
'''

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#loading in data
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

#checking data is loaded
print(data.head(5))

#splitting the data into features and target variable
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# making the model
logreg = LogisticRegression()

# fitting the model
logreg.fit(X_train, y_train)

# predicting the test set
y_pred = logreg.predict(X_test)

# Evaluating model
print("Confusion Matrix for Logistic Regression: ")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

print("Interpretation of Regular SGD Confusion Matrix: ")
print(" True Positives (TP): ", cm[1, 1])
print(" True Negatives (TN): ", cm[0, 0])
print(" False Positives (FP): ", cm[0, 1])
print(" False Negatives (FN): ", cm[1, 0])

print("Classification Report for Logistic Regression: ")
print(classification_report(y_test, y_pred, labels=[0, 1]))

logreg_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression: ", logreg_accuracy)

# looking at feature importance, or the Coefficients of the LogReg
print("Logistic Regression Coefficients: ")
print(logreg.coef_)