import pandas as pd

# Read in the data
data = pd.read_csv('skills.csv')

# Count the number of occurrences of each skill
skill_counts = data['skills'].value_counts()

# Compute the cumulative sum of skill counts
cumulative_counts = skill_counts.cumsum()

# Compute the total number of rows
total_rows = data.shape[0]

# Find the index of the last skill that covers at least 80% of the data
last_skill_index = (cumulative_counts / total_rows).searchsorted(0.6, side='right')

# Get the list of top skills that cover at least 80% of the data
top_skills = skill_counts.index[:last_skill_index]

# Create a new DataFrame with one row per unique user_id
unique_users = data.drop_duplicates(subset='user_id', keep='first').set_index('user_id')

# Add columns for each top skill and initialize with 0
for skill in top_skills:
    unique_users[skill] = 0

# Set values to 1 for each user/skill combination
for i, row in data.iterrows():
    if row['skills'] in top_skills:
        unique_users.at[row['user_id'], row['skills']] = 1

# Save the results to a new file
unique_users.to_csv('skills_processed_data.csv')
