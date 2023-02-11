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

#Education Degree Analysis

import pandas as pd
from variables import BACHELORS_DEGREE,ASSOCIATE_DEGREE,MASTER_DEGREE,DOCTORATE_DEGREE

df_education=pd.read_pickle("../data/226k_processed_metadata_august_title_lowered_rank_splitted_monthworked_score_industry_sector_rank_added.pkl")
thisdict = {'Associate':0, 'Bachelor':0, 'Master':0, 'Doctorate':0,}
other_degrees = dict()

def analyzer(row):
    if row['education'] != None:
        for i in range(len(row['education'])):
            global thisdict
            local_dict = {'Associate':0,'Bachelor':0,'Master':0,'Doctorate':0,}

            row['num_associate'] = 0
            row['num_bachelor'] = 0
            row['num_master'] = 0
            row['num_doctorate'] = 0

            deg=row['education'][i]['degree']
            row['education'][i]['degree'] = str(row['education'][i]['degree']).replace("'","")
            words = str(row['education'][i]['degree']).split()
            if len(set(words).intersection(set(BACHELORS_DEGREE))) != 0:
                thisdict['Bachelor'] += 1
                local_dict['Bachelor'] += 1
                row['num_bachelor'] = 1
            elif len(set(words).intersection(set(MASTER_DEGREE))) != 0:
                thisdict['Master'] += 1
                local_dict['Master'] += 1
                row['num_master'] = 1
            elif len(set(words).intersection(set(DOCTORATE_DEGREE))) != 0:
                thisdict['Doctorate'] += 1
                local_dict['Doctorate'] += 1
                row['num_doctorate'] = 1
            elif len(set(words).intersection(set(ASSOCIATE_DEGREE))) != 0:
                thisdict['Associate'] += 1
                local_dict['Associate'] += 1
                row['num_associate'] = 1
            else:
                if deg in other_degrees:
                    other_degrees[deg] += 1
                else:
                    other_degrees[deg] = 1

            row['education_distribution'] = local_dict
    return row


df_education = df_education.apply(lambda row: analyzer(row),axis=1)
