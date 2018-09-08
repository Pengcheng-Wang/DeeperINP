"""
This script is used to run several baselines for player score prediction in CI-RL dataset.
Created by pwang8, on Sept 8, 2018.
"""
import numpy as np
from pandas import read_csv
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

score_data = read_csv('../../userModelTrained/userStateRewardOnly.csv')
# shape: (402, 22)
score_data_X = score_data.iloc[:, 0:21]
# shape: (402, 21)
score_data_X_norm = (score_data - score_data.mean()) / (score_data.max() - score_data.min())
score_data_Y = score_data.iloc[:, 21:]
# shape: (402, 1)

scoring = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted']   # evaluation metrics

# SVM
score_clf = SVC(kernel='linear', C=0.1)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredSVM_linear_c.1.txt', 'a'))

score_clf = SVC(kernel='linear', C=0.5)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredSVM_linear_c.5.txt', 'a'))

score_clf = SVC(kernel='linear', C=1.0)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredSVM_linear_c1.txt', 'a'))

# RF
from sklearn.ensemble import RandomForestClassifier
score_clf = RandomForestClassifier(max_depth=5, n_estimators=5)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredRF_dep5_nes5.txt', 'a'))

score_clf = RandomForestClassifier(max_depth=5, n_estimators=10)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredRF_dep5_nes10.txt', 'a'))

score_clf = RandomForestClassifier(max_depth=10, n_estimators=5)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredRF_dep10_nes5.txt', 'a'))

score_clf = RandomForestClassifier(max_depth=10, n_estimators=10)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredRF_dep10_nes10.txt', 'a'))

# GB
from sklearn.ensemble import GradientBoostingClassifier
score_clf = GradientBoostingClassifier(max_depth=5, n_estimators=5)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredGB_dep5_nes5.txt', 'a'))

score_clf = GradientBoostingClassifier(max_depth=5, n_estimators=10)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredGB_dep5_nes10.txt', 'a'))

score_clf = GradientBoostingClassifier(max_depth=10, n_estimators=5)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredGB_dep10_nes5.txt', 'a'))

score_clf = GradientBoostingClassifier(max_depth=10, n_estimators=10)
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredGB_dep10_nes10.txt', 'a'))

# NB
from sklearn.naive_bayes import GaussianNB
score_clf = GaussianNB()
scores = cross_validate(score_clf, score_data_X_norm, score_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/scorePredNB.txt', 'a'))
