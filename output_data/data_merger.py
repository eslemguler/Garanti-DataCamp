import pandas as pd

# read each CSV file
test = pd.read_csv('test_users.csv')
train = pd.read_csv('train_users.csv')

edu = pd.read_csv('edu_ops.csv')
exp = pd.read_csv('experience_ops.csv')
lang = pd.read_csv('language_ops.csv')
skill = pd.read_csv('skill_ops_140_col.csv')

# merge the CSV files based on user_id
merged_df = pd.merge(train,edu, on='user_id', how='left')
merged_df = pd.merge(merged_df, exp, on='user_id', how='left')
merged_df = pd.merge(merged_df, lang, on='user_id', how='left')
merged_df = pd.merge(merged_df, skill, on='user_id', how='left')

# fill missing values with appropriate method
merged_df.fillna(method='ffill', inplace=True)  # forward fill method

# save the merged data to a new CSV file
merged_df.to_csv('merged_data_train.csv', index=False)

