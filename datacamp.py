import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys

from variables import BACHELORS_DEGREE,ASSOCIATE_DEGREE,MASTER_DEGREE,DOCTORATE_DEGREE
edu = pd.read_csv('education.csv')
language = pd.read_csv('languages.csv')
skills = pd.read_csv('skills.csv')
submission = pd.read_csv('submission.csv')
test_users = pd.read_csv('test_users.csv')
train_users = pd.read_csv('train_users.csv')
exp = pd.read_csv('work_experiences.csv')
