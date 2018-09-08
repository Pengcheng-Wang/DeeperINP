"""
This script is used to run a Naive Bayes baseline for player action prediction in CI-RL dataset.
Created by pwang8, on Sept 8, 2018.
"""
import numpy as np
from pandas import read_csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

act_data = read_csv('../../userModelTrained/userStateActOnly.csv')
# shape: (16313, 22)
act_data_X = act_data.iloc[:, 0:21]
# shape: (16313, 21)
act_data_X_norm = (act_data_X - act_data_X.mean()) / (act_data_X.max() - act_data_X.min())
act_data_Y = act_data.iloc[:, 21:]
# shape: (16313, 1)

scoring = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted']
act_clf = GaussianNB()
scores = cross_validate(act_clf, act_data_X_norm, act_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/actPredNB.txt', 'a'))