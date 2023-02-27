import pandas as pd

# read each CSV file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# # Load the data into a Pandas dataframe
# df = pd.read_csv('study_ops.csv')

# # Group the data by user_id and sum the values for each column
# df_grouped = df.groupby('user_id').sum()

# # Reset the index to make user_id a regular column
# df_grouped = df_grouped.reset_index()

# # Display the resulting dataframe
# df_grouped.to_csv('study_ops_processed.csv', index=False)


# ----------------------------------------------------------------
# # Concatenate the dataframes to get a single list of categories
# categories = pd.concat([test['location'], train['location']], ignore_index=True)

# # Create a categorical variable using the categories
# cat_var = pd.Categorical(categories)

# # Encode the column in both dataframes using the categorical variable
# test['location_encoded'] = cat_var.codes[:len(test)]
# train['location_encoded'] = cat_var.codes[-len(train):]

# test.to_csv('test_merged.csv')
# train.to_csv('train_merged.csv')

# ----------------------------------------------------------------
edu = pd.read_csv('edu_ops.csv')
exp = pd.read_csv('experience_ops.csv')
lang = pd.read_csv('language_ops.csv')
skill = pd.read_csv('skill_ops_180_col.csv')
study = pd.read_csv('study_ops_processed.csv')

# merge the CSV files based on user_id
# merged_df = pd.merge(train,edu, on='user_id', how='left')
# merged_df = pd.merge(merged_df, exp, on='user_id', how='left')
# merged_df = pd.merge(merged_df, lang, on='user_id', how='left')
# merged_df = pd.merge(merged_df, skill, on='user_id', how='left')
merged_df = pd.merge(train, study, on='user_id', how='left')

# fill missing values with appropriate method
merged_df.fillna(method='bfill', inplace=True) 

# save the merged data to a new CSV file
merged_df.to_csv('train_merged_w_study.csv', index=False)

