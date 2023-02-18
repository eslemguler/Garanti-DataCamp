import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from variables import BACHELORS_DEGREE,ASSOCIATE_DEGREE,MASTER_DEGREE,DOCTORATE_DEGREE
edu = pd.read_csv('education.csv')
edu = edu[['user_id','school_name','degree']]
language = pd.read_csv('languages.csv')
skills = pd.read_csv('skills.csv')
submission = pd.read_csv('submission.csv')
test_users = pd.read_csv('test_users.csv')
train_users = pd.read_csv('train_users.csv')
exp = pd.read_csv('work_experiences.csv')

#Education Degree Analysis
from variables import BACHELORS_DEGREE,ASSOCIATE_DEGREE,MASTER_DEGREE,DOCTORATE_DEGREE

def education_analyzer(row):
    local_dict = {'Associate':0,'Bachelor':0,'Master':0,'Doctorate':0,}

    row['num_associate'] = 0
    row['num_bachelor'] = 0
    row['num_master'] = 0
    row['num_doctorate'] = 0

    row['degree'] = str(row['degree']).replace("'","")
    words = str(row['degree']).split()

    if len(set(words).intersection(set(MASTER_DEGREE))) != 0:
        local_dict['Master'] += 1
        row['num_master'] = 1
    elif len(set(words).intersection(set(DOCTORATE_DEGREE))) != 0:
        local_dict['Doctorate'] += 1
        row['num_doctorate'] = 1
    elif len(set(words).intersection(set(ASSOCIATE_DEGREE))) != 0:
        local_dict['Associate'] += 1
        row['num_associate'] = 1
    elif len(set(words).intersection(set(BACHELORS_DEGREE))) != 0 or len(set(str(row['school_name']).split()).intersection(set(BACHELORS_DEGREE)))!= 0:
        local_dict['Bachelor'] += 1
        row['num_bachelor'] = 1

    # row['education_distribution'] = local_dict
    return row

edu = edu.apply(lambda row: education_analyzer(row),axis=1)
edu = edu[['user_id','num_associate','num_master','num_bachelor','num_doctorate']]
agg_functions = {'user_id': 'first','num_associate': 'sum', 'num_bachelor': 'sum','num_master': 'sum', 'num_doctorate': 'sum'}
edu = edu.groupby(edu['user_id']).aggregate(agg_functions)
print(edu)
edu.to_csv('edu.csv', index=False)
exit()
