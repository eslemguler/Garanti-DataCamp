import pandas as pd

# Create a sample dataframe
df = pd.read_csv('work_experiences.csv')

# Get the number of companies each user worked for
company_counts = df.groupby('user_id')['company_id'].nunique()

# Get the smallest start date and biggest start date for each user
start_dates = df.groupby(['user_id', 'company_id'])['year'].min()
end_dates = df.groupby(['user_id', 'company_id'])['year'].max()
smallest_start_dates = start_dates.groupby('user_id').min()
biggest_end_dates = end_dates.groupby('user_id').max()

# Calculate the approximate working time for each user
working_times = (biggest_end_dates - smallest_start_dates) / company_counts

# Add the new columns to the original dataframe
df['company_count'] = df['user_id'].map(company_counts)
df['working_time'] = df['user_id'].map(working_times)

df.to_csv('experience_ops.csv')