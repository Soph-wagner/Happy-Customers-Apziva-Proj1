'''
Author: Sophia Wagner
Date: 6/17/2024
Description: Recursive Feature Elimination for ACME Happiness Survey Data
'''

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import MinMaxScaler

# importing dataset
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

# checking data is loaded
print(data.head(5))

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# creating all 3 models to compare at estimators 
SGD = SGDClassifier(loss='hinge')
logreg = LogisticRegression()
rfc = RandomForestClassifier(n_estimators=100, random_state=224)
### also creating a MinMax scaler for the RFECV
minmax = MinMaxScaler()
X_scale_minmax = minmax.fit_transform(X)
mm_X_train, mm_X_test, mm_y_train, mm_y_test = train_test_split(X_scale_minmax, y, test_size = 0.2, random_state=0)

# creating RFECV for each model & computing the accuracy 
# using a StratifiedKFold of 10
### SGD
rfecv_SGD = RFECV(estimator=SGD, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv_SGD.fit(X_train, y_train)
### Logistic Regression
rfecv_lr = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv_lr.fit(X_train, y_train)
### Random Forest
rfecv_rfc = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv_rfc.fit(X_train, y_train)
### MinMax SGD
rfecv_mm_SGD = RFECV(estimator=SGD, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv_mm_SGD.fit(mm_X_train, mm_y_train)

# print optimal number of features for each model
print("Optimal number of features for SGD: %d" % rfecv_SGD.n_features_)
print("Optimal number of features for Logistic Regression: %d" % rfecv_lr.n_features_)
print("Optimal number of features for Random Forest: %d" % rfecv_rfc.n_features_)
print("Optimal number of features for MinMax SGD: %d" % rfecv_mm_SGD.n_features_)

# print feature ranking
print("Feature ranking for SGD: ", rfecv_SGD.ranking_)
print("Feature ranking for Logistic Regression: ", rfecv_lr.ranking_)
print("Feature ranking for Random Forest: ", rfecv_rfc.ranking_)
print("Feature ranking for MinMax SGD: ", rfecv_mm_SGD.ranking_)


###### THIS PLOT DOES NOT WORK #############
# plot number of features vs. accuracy scores
plt.figure()
plt.xlabel("Number of selected features")
plt.ylabel("accuracy score (num correct classifications)")
plt.title("Recursive Feature Elimination with Cross Validation")
plt.plot(range(1, len(rfecv_SGD.grid_scores_) + 1), rfecv_SGD.grid_scores_, label='SGD')
plt.plot(range(1, len(rfecv_lr.grid_scores_) + 1), rfecv_lr.grid_scores_, label='Logistic Regression')
plt.plot(range(1, len(rfecv_rfc.grid_scores_) + 1), rfecv_rfc.grid_scores_, label='Random Forest')
plt.plot(range(1, len(rfecv_mm_SGD.grid_scores_) + 1), rfecv_mm_SGD.grid_scores_, label='MinMax SGD')
plt.legend()
plt.show()