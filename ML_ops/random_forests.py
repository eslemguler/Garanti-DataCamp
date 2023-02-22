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

# Split into features and target
X = train_data.drop('moved_after_2019', axis=1)
y = train_data['moved_after_2019']

# Define the preprocessing steps for the different types of columns
numeric_features = ["num_associate", "num_bachelor", "num_master", "num_doctorate", "company_count", "working_time"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

categorical_features = ["industry", "location"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

skill_features = list(X.columns[11:])
skill_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

# Combine the preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("skill", skill_transformer, skill_features)
    ]
)

# Define the classifier to use
clf = RandomForestClassifier()

# Combine the preprocessor and classifier in a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

# Define the hyperparameters to tune using grid search
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20]
}

# Use grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)

# Make predictions on the test data
y_pred = grid_search.predict(test_data)

# Create a submission file
submission_df = pd.DataFrame({'user_id': test_data['user_id'], 'churn': y_pred})
submission_df.to_csv('submission.csv', index=False)
