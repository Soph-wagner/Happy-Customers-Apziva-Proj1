'''
Author: Sophia Wagner
Date: 7/01/2024
Description: Creating a final model for the ACME Happiness Survey Data
             Going with a Random Forest Classifier model
'''

# importing libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import joblib

'''
### DESCRIPTION OF THE MODEL ###
-- why this model: Random Forest Classifier is the model that performed
    the best compared to SGD Classifier and Logistic Regression

-- feature set: {'X1', 'X3', 'X6'}
-------- 'X1' question: my order was delivered on time
-------- 'X3' question: I ordered everything I wanted to order
-------- 'X6' question: the app makes ordering easy for me

'''

# Importing the dataset
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

# making sure data is loaded
print(data.head(5))

X = data[['X1', 'X3', 'X6']].values

y = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
print("Training set size: ", X_train.shape)
print("Test set size: ", X_test.shape)

#print the counts of 1 and 0 in the test set
print("Counts of 1 and 0 in the test set: ")
print(pd.Series(y_test).value_counts())

# creating the Random Forest model
rf_model = RandomForestClassifier(n_estimators=50)

# fitting the model
rf_model.fit(X_train, y_train)

# predicting the test set results
y_pred = rf_model.predict(X_test)

#print classification report
print("Classification Report for Random Forest: ")
print(classification_report(y_test, y_pred, labels=[0, 1]))

#Evaluating model
print("Confusion Matrix for Random Forest: ")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

print("Interpretation of Confusion Matrix: ")
print(" True Positives (TP): ", cm[1, 1])
print(" True Negatives (TN): ", cm[0, 0])
print(" False Positives (FP): ", cm[0, 1])
print(" False Negatives (FN): ", cm[1, 0])

rf_accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy: ", rf_accuracy)

'''
#now Looking at Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=['X1', 'X3', 'X6']).sort_values(ascending=False)
print("Feature Importance: ", feature_importance)
''' 

# SAVING THE WEIGHTS OF THIS MODEL !!! 
#### SHE FINALLY DID IT! 
import pickle 
filename = 'finalized3_rfc_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))
print("pickle Model Saved!")

#using joblib to save the model as well (not sure which is better yet)
filename = 'finalized3_rfc_model_joblib.sav'
joblib.dump(rf_model, filename)
print("joblib Model Saved!")
