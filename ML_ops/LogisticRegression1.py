#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data
train_data = pd.read_csv("merged_data_train.csv")
test_data = pd.read_csv("merged_data_test.csv")


# In[3]:


# Split into features and target
X = train_data.drop('moved_after_2019', axis=1)
y = train_data['moved_after_2019']


# In[4]:


# Define the preprocessing steps for the different types of columns
numeric_features = ["num_associate", "num_bachelor", "num_master", "num_doctorate", "company_count", "working_time"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


# In[5]:


categorical_features = ["industry", "location"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])


# In[6]:


skill_features = list(X.columns[11:])
skill_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

# Combine the preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("skill", skill_transformer, skill_features)
    ]
)


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[8]:


logisticRegr = LogisticRegression()


# In[9]:


pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", logisticRegr)])


# In[10]:


# Define the hyperparameters to tune using grid search
param_grid = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [0.1, 1.0, 10.0]
}      


# In[11]:


# Use grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)


# In[12]:


# Make predictions on the test data
y_pred = grid_search.predict(test_data)

# Create a submission file
submission_df = pd.DataFrame({'user_id': test_data['user_id'], 'moved_after_2019': y_pred})
submission_df.to_csv('submission_rf.csv', index=False)


# In[ ]:




