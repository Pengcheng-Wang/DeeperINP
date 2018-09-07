"""
This script is used to run a SVM baseline for player action prediction in CI-RL dataset.
Created by pwang8, on Sept 7, 2018.
"""
import numpy as np
from pandas import read_csv
from sklearn.svm import SVC

act_data = read_csv('../userModelTrained/userStateActOnly.csv')
print(act_data.shape)
print(act_data,loc[[0]])
