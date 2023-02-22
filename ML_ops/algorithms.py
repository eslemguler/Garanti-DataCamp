import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load the data
data = pd.read_csv("your_data.csv")

# Split into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for numerical features
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Define the column transformer for categorical features
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Define the column transformer for all features
preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

# Define the classifiers to use
classifiers = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    GaussianNB()
]

# Define the hyperparameters to tune using grid search
param_grids = [
    {
        'classifier': [LogisticRegression()],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.1, 1.0, 10.0],
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6, 10],
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf'],
    },
    {
        'classifier': [GaussianNB()]
    }
]

# Use cross-validation to find the best classifier and hyperparameters
best_score = 0
best_classifier = None
for clf, param_grid in zip(classifiers, param_grids):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"Classifier: {clf.__class__.__name__}, Best hyperparameters: {grid_search.best_params_}, Mean cross-validation score: {mean_score}")
    if mean_score > best_score:
        best_score = mean_score
        best_classifier = grid_search.best_estimator_

# Evaluate the best classifier on the test data
y_pred = best_classifier.predict(X_test)
print(classification_report(y_test, y_pred))
