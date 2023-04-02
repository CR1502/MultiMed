# import the necessary libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense

# loading the data from sklearn
bcancer_data = sklearn.datasets.load_breast_cancer()
print(bcancer_data)

# loading the data to a data frame
df = pd.DataFrame(bcancer_data.data, columns=bcancer_data.feature_names)

# Displaying a portion of the data
print(df.head())

df.rename(columns={'mean radius': 'mean_radius', 'mean texture': 'mean_texture', 'mean perimeter': 'mean_perimeter',
                   'mean area': 'mean_area', 'mean smoothness': 'mean_smoothness',
                   'mean compactness': 'mean_compactness', 'mean concavity': 'mean_concavity',
                   'mean concave points': 'mean_concave_points', 'mean symmetry': 'mean_symmetry',
                   'mean fractal dimension': 'mean_fractal_dimension', 'radius error': 'radius_error',
                   'texture error': 'texture_error', 'perimeter error': 'perimeter_error', 'area error': 'area_error',
                   'smoothness error': 'smoothness_error', 'compactness error': 'compactness_error',
                   'concavity error': 'concavity_error', 'concave points error': 'concave_points_error',
                   'symmetry error': 'symmetry_error', 'fractal dimension error': 'fractal_dimension_error',
                   'worst radius': 'worst_radius', 'worst texture': 'worst_texture',
                   'worst perimeter': 'worst_perimeter', 'worst area': 'worst_area',
                   'worst smoothness': 'worst_smoothness', 'worst compactness': 'worst_compactness',
                   'worst concavity': 'worst_concavity', 'worst concave points': 'worst_concave_points',
                   'worst symmetry': 'worst_symmetry', 'worst fractal dimension': 'worst_fractal_dimension'}, inplace=True)

# adding the 'target' column to the data frame
df['label'] = bcancer_data.target

# Displaying a portion of the data
print(df.head())

# Displaying information about the data
print(df.info())

# Checking if there are any null elements in the dataset
print(df.isnull().sum())

# Describing the data
print(df.describe())

# Checking the distribution of Target Variable
df['label'].value_counts()
# 1 --> Benign
# 0 --> Malignant

df.groupby('label').mean()

# Splitting the features and the targets
X = df.drop(columns='label', axis=1)
y = df['label']

print(X)

print(y)

# Splitting the data into Training data and Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Making an ANN model
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, shuffle=False)

# Evaluating the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Building a Predictive System
input_data = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058,
              23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773,
              0.239, 0.1288, 0.2977, 0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Breast cancer is Malignant')
else:
    print('The Breast Cancer is Benign')

# Saving the trained model
import pickle
import joblib
filename = 'breast_cancer_model_for_web.pkl'
joblib.dump(prediction, filename)