import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_mentors=pd.read_pickle("../data/224k_processed_metadata_old.pkl")
df_skills=df_mentors[['skills']].copy()
unique_skills_dict = dict()
removed_keys = pd.Series()

def unique_skills(row):
    '''
    Does:
        Finds the number of each skill and lists them with the counts.
            First check is : if the skill is in unique_skills_dict or not
            if it is:
                    add one to the count
            if its not:
                create the key and set it as 1
    ----------
    Parameters
    ----------
    row: df_skills
    unique_skills_dict: dictionary of unique skills
    Returns
    -------
    unique_skills_dict
    '''
    for i in range(len(row['skills'])):

        if row['skills'][i] in unique_skills_dict.keys() :
            unique_skills_dict[row['skills'][i]] += 1
        else:
            unique_skills_dict[row['skills'][i]] = 1
    return unique_skills_dict

df_skills = df_skills.apply(lambda row: unique_skills(row),axis=1)
df = pd.DataFrame(list(unique_skills_dict.items()), columns = ['skills', 'count'])
df_skills = df['skills'].copy()

vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(df_skills)
#vectorizes the strings in df_skills so that we can use it to find cos sim
cos_sim = cosine_similarity(vector,vector)
#finds cosine similarity

for i in range(len(cos_sim[0])):
    for j in range(len(cos_sim[0])):
        if cos_sim[i][j] > 0.6 and cos_sim[i][j]<1:
            #checks if the cosine_similarity is bigger than 0.6 and smaller than 1
            if df['count'][i] > df['count'][j]:
                #checks the frequency of the two compared strings, bigger one stays smaller eliminates
                removed_keys = pd.concat([removed_keys,pd.Series(df_skills[j])], ignore_index=True)
            elif df['count'][i] < df['count'][j]:
                removed_keys = pd.concat([removed_keys,pd.Series(df_skills[i])], ignore_index=True)
            else:
                pass
        else:
            pass

trophy = pd.concat([df_skills, removed_keys]).drop_duplicates(keep=False)
#does the calculation df_skills - removed_keys to find the clustered 'unique' skills by dropping the duplicates
print(trophy)
