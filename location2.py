
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('location.csv')

from cities import Istanbul,Ankara,Izmir,Bursa



def location_analyzer(row):
    local_dict = {'numIstanbul':0,'numAnkara':0,'numIzmir':0,'numBursa':0,}

    row['Istanbul'] = 0
    row['Ankara'] = 0
    row['Izmir'] = 0
    row['Bursa'] = 0

    row['column2'] = str(row['column2']).replace("[.,/;:","")
    words = str(row['column2']).split()

    if len(set(words).intersection(set(Istanbul))) != 0:
     #     local_dict['numIstanbul'] += 1
         row['Istanbul'] = 1
    if len(set(words).intersection(set(Ankara))) != 0:
     #     local_dict['numAnkara'] += 1
         row['Ankara'] = 1
    if len(set(words).intersection(set(Izmir))) != 0:
     #     local_dict['numIzmir'] += 1
         row['Izmir'] = 1
    if len(set(words).intersection(set(Bursa))) != 0:
     #     local_dict['numBursa'] += 1
         row['Bursa'] = 1


    # row['education_distribution'] = local_dict
    return row

  
df = df.apply(lambda row: location_analyzer(row),axis=1)
df.to_csv('df.csv')

exit()