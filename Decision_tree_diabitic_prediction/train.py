import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# load the dataset
df = pd.read_csv('data/diabetes (1).csv')
print(df.head())
print(df.info())

# define lables and features
X = df.drop('Outcome',axis=1)
Y = df['Outcome']

# split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# training the model
model = RandomForestClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5)
model.fit(x_train,y_train)

# prediction
prediction = model.predict(x_test)
print(prediction)

# accuracy
accuracy = accuracy_score(y_test, prediction)
print('Accuracy:', accuracy*100)

# saving the trained model
joblib.dump(model,'model/model.pkl')


# Using the GridSearchCV for hyperparameter tuning

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
#
# # Load the dataset
# df = pd.read_csv('data/diabetes (1).csv')
# print("First 5 rows of the dataset:")
# print(df.head())
#
# # Display dataset information
# print("\nDataset Information:")
# print(df.info())
#
# # Define labels and features
# X = df.drop('Outcome', axis=1)
# Y = df['Outcome']
#
# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # Define hyperparameter search space
# param_grid = {
#     'n_estimators': [50, 100, 200],  # Number of trees
#     'max_depth': [3, 5, 10, None],   # Tree depth
#     'min_samples_split': [2, 5, 10], # Minimum samples required to split
#     'min_samples_leaf': [1, 3, 5],   # Minimum samples per leaf
#     'criterion': ['gini', 'entropy'] # Test both splitting criteria
# }
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#
# # Train the model with different hyperparameters
# print("\nRunning GridSearchCV to find the best hyperparameters...")
# grid_search.fit(x_train, y_train)
#
# # Check if GridSearchCV found the best parameters
# if grid_search.best_params_:
#     print("\nBest Parameters:", grid_search.best_params_)
# else:
#     print("\nGrid search did not find any best parameters.")
#
# # Train Random Forest with the best parameters
# best_model = grid_search.best_estimator_
# print("\nTraining the model with the best parameters...")
# best_model.fit(x_train, y_train)
# print("Model training complete.")
#
# # Make predictions
# print("\nMaking predictions on the test set...")
# prediction = best_model.predict(x_test)
#
# # Evaluate accuracy
# accuracy = accuracy_score(y_test, prediction)
# print(f'\nOptimized Accuracy: {accuracy * 100:.2f}%')
