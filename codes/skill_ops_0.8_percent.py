import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('skills.csv')

# Compute the total number of users
n_users = df['user_id'].nunique()

# Compute the frequency of each skill
skill_counts = df['skills'].value_counts(normalize=True)

# Get the skills that occur in more than 1% of the users
popular_skills = skill_counts[skill_counts > 0.001].index

# Create a new dataframe with one row for each unique user
user_df = df.drop_duplicates(subset='user_id').copy()
user_df.set_index('user_id', inplace=True)

# Create a new column for each popular skill
for skill in popular_skills:
    user_df[skill] = 0

# Set the value to 1 for each user that has the skill
for i, row in df.iterrows():
    if row['skills'] in popular_skills:
        user_df.loc[row['user_id'], row['skills']] = 1

# Print the resulting dataframe
print(user_df)
user_df.to_csv('skill_ops.csv')
