import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('work_experiences.csv')

# Extract the desired column from the DataFrame
c2 = df['location']
c1 = df['user_id']

final_df = pd.DataFrame({'column1': c1, 'column2': c2})

final_df.to_csv('location.csv', index=False)
exit()