'''
Author: Sophia Wagner
Date: May 20th, 2024
Description: This file contains the customer satisfaction prediction model for Happy Customers Apziva project 1.
'''

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

# quick look at the data
print(data.head())


# Checking for missing values
#print("Missing values:")
#print(data.isnull().sum())
# There are no missing values in the dataset

#creating a model that predicts if a customer is happy or not based on the answers they provided in the survey
#splitting the data into features and target variable
### entries in first column are target variable, all other columns are features
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

#print(X)
#print(y)

# Splitting the dataset into the Training set and Test set
# 127 total entries, 80% for training, 20% for testing so 101 training entries, 26 testing entries
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#our target variable is categorical, so we will use a classification model
#I am going to try out a few different classification models

# Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)
LR_y_pred = LR_model.predict(X_test)

LR_accurary = accuracy_score(y_test, LR_y_pred)

print("Logistic Regression Accuracy: ", LR_accurary)


'''

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for Logistic Regression: ")
print(cm)
print("printing y_test and y_pred")
print(y_test)
print(y_pred)

'''

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)
DT_y_pred = DT_model.predict(X_test)

DT_accurary = accuracy_score(y_test, DT_y_pred)

print("Decision Tree Accuracy: ", DT_accurary)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
SVM_model = SVC()
SVM_model.fit(X_train, y_train)
SVM_y_pred = SVM_model.predict(X_test)

SVM_accurary = accuracy_score(y_test, SVM_y_pred)

print("SVM Accuracy: ", SVM_accurary)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)
RF_y_pred = RF_model.predict(X_test)

RF_accurary = accuracy_score(y_test, RF_y_pred)

print("Random Forest Accuracy: ", RF_accurary)

# K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)
KNN_y_pred = KNN_model.predict(X_test)

KNN_accurary = accuracy_score(y_test, KNN_y_pred)

print("K-NN Accuracy: ", KNN_accurary)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
NB_y_pred = NB_model.predict(X_test)

NB_accurary = accuracy_score(y_test, NB_y_pred)

print("Naive Bayes Accuracy: ", NB_accurary)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
GB_model = GradientBoostingClassifier()
GB_model.fit(X_train, y_train)
GB_y_pred = GB_model.predict(X_test)

GB_accurary = accuracy_score(y_test, GB_y_pred)

print("Gradient Boosting Accuracy: ", GB_accurary)

# Evaluating the models ? which is best?
'''
All of the models are performing similarly, w/ accuracies ranging from 0.46 to 0.61

I have yet to check out their confusionn matrixes

I will do a bit more research to more confidently choose which model arcitecture is best
Then, I will try to tune the hyperparameters of the selected model to see if I can improve the accuracy

'''



# Also trying out a neural network model (MLP Classifier in sklearn)




#bonus goal: testing all X1 through X6 