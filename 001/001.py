# import sklearn
# # print(sklearn.__version__)
# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression

# categorical_features = ['company', 'name', 'fuel_type']
# numeric_features = ['year', 'kms_driven']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#         ('num', 'passthrough', numeric_features)
#     ])

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load your data
car = pd.read_csv('Cleaned_Car_data.csv')

# Define features and target
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['name', 'company', 'fuel_type']
numeric_features = ['year', 'kms_driven']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
with open('LinearRegressionModel.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
