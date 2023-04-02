# import the necessary libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Reading the data
parkinson_data = pd.read_csv('parkinsons.csv')

# Displaying a portion of the data
print(parkinson_data.head())

# Displaying information about the data
print(parkinson_data.info())

# Checking if there are any null elements in the dataset
print(parkinson_data.isnull().sum())

# Describing the data
print(parkinson_data.describe())

# Printing the count of the target values
print(parkinson_data['status'].value_counts())
# 1 --> Parkinson's Positive
# 0 --> Healthy

# grouping the data bas3ed on the target variable
print(parkinson_data.groupby('status').mean())

# Splitting the features and the targets
X = parkinson_data.drop(columns=['name', 'status'], axis=1)
y = parkinson_data['status']

print(X)

print(y)

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

# Splitting the data into Training data and Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

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
input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563,
              0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have Parkinson's Disease")

else:
    print("The Person has Parkinson's")

# Saving the trained model
filename = 'parkinson_s_model_for_web.h5'
joblib.dump(model, filename)
