import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data
data = pd.read_csv("merged_data_train.csv")
test_data = pd.read_csv("merged_data_test.csv")
submission = pd.read_csv('submission_ex.csv')

# Split into features and target
X = data.drop('moved_after_2019', axis=1)
y = data['moved_after_2019']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
clf = xgb.XGBClassifier()

# Combine the preprocessor and classifier in a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


# Define the hyperparameters to tune using grid search
param_grid = {
    "classifier__n_estimators": [200, 500, 1000],
    "classifier__max_depth": [10, 5, 12],
    "classifier__learning_rate": [0.1, 0.15]
}

# Use cross-validation to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the model using cross-validation
scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Apply the same preprocessing steps to the test data
X_test_transformed = preprocessor.transform(test_data)

# Make predictions on the test data
y_pred = grid_search.predict(X_test_transformed)

# Print the classification report
print(classification_report(y, y_pred))

# Create a submission file
submission_df = pd.DataFrame({'user_id': submission['user_id'], 'moved_after_2019': y_pred})
submission_df.to_csv('submission.csv', index=False)
