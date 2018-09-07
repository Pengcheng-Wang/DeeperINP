"""
This script is used to run a Random Forest baseline for player action prediction in CI-RL dataset.
Created by pwang8, on Sept 7, 2018.
"""
import numpy as np
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

act_data = read_csv('../../userModelTrained/userStateActOnly.csv')
# shape: (16313, 22)
act_data_X = act_data.iloc[:, 0:21]
# shape: (16313, 21)
act_data_X_norm = (act_data_X - act_data_X.mean()) / (act_data_X.max() - act_data_X.min())
act_data_Y = act_data.iloc[:, 21:]
# shape: (16313, 1)

scoring = ['accuracy', 'f1_macro', 'f1_micro']
act_clf = RandomForestClassifier(max_depth=3, n_estimators=10)
scores = cross_validate(act_clf, act_data_X_norm, act_data_Y, scoring=scoring, cv=5, return_train_score=False, verbose=True)

print(scores, file=open('../../userModelTrained/actPredRF_dep3_ne10.txt', 'a'))
