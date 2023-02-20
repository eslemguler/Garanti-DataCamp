import pandas as pd
from fuzzywuzzy import fuzz

# # Load the dataset
# df = pd.read_csv('languages.csv')

# # Define the target values
# target_values = ['Almanca', 'Arabic', 'Arapça', 'Azerbaijani', 'Azerice', 'Bulgarca', 'Bulgarian',
#                  'Chinese', 'Dutch', 'English', 'english', 'Fransızca', 'French', 'German',
#                  'ingilizce', 'Italian', 'Japanese', 'Japonca', 'Korean', 'Korece', 'Kürtçe',
#                  'Lehçe', 'Persian', 'Russian', 'Rusça', 'Spanish', 'Turkish', 'Türkçe',
#                  'Urdu', 'Çince', 'İngilizce', 'İspanyolca', 'İtalyanca']

# # Create a function to calculate the similarity score for each language
# def get_similarity_score(language):
#     for target in target_values:
#         score = fuzz.token_set_ratio(language.lower(), target.lower())
#         if score >= 90:  # Set the threshold for a match
#             return target
#     return 'Other'  # Assign 'Other' if no match is found

# # Apply the function to the 'language' column
# df['similar_language'] = df['language'].apply(get_similarity_score)

# df.to_csv('fuzzywuzzy.csv')
# # Preview the new column
# print(df['similar_language'].value_counts())

# Read in the data
df = pd.read_csv('fuzzywuzzy.csv')

# Get a list of unique languages
languages = df['language'].unique()

# Create a new DataFrame with one row per user ID
new_df = pd.DataFrame({'user_id': df['user_id'].unique()})

# Add a new column for each language and set the proficiency as the value
for language in languages:
    # Filter the original DataFrame to get only rows with the current language
    language_df = df[df['language'] == language]
    # Merge the filtered DataFrame with the new DataFrame using the user ID
    merged_df = pd.merge(new_df, language_df[['user_id', 'proficiency']], on='user_id', how='left')
    # Rename the proficiency column to the name of the current language
    renamed_df = merged_df.rename(columns={'proficiency': language})
    # Fill any missing values with 0
    filled_df = renamed_df.fillna(0)
    # Replace the old DataFrame with the new one
    new_df = filled_df

print(new_df)
new_df.to_csv('new_df_fuzzywuzzy.csv', index=False)