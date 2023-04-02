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
diabetes_data = pd.read_csv('diabetes.csv')

# Displaying a portion of the data
print(diabetes_data.head())

# Displaying information about the data
print(diabetes_data.info())

# Checking if there are any null elements in the dataset
print(diabetes_data.isnull().sum())

# Describing the data
print(diabetes_data.describe())

# Printing the count of the target values
print(diabetes_data['Target'].value_counts())
# 1 --> Defective Heart
# 0 --> Healthy Heart

# Splitting the features and the targets
X = diabetes_data.drop(columns='Target', axis=1)
y = diabetes_data['Target']

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
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = model.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

# Saving the trained model
filename = 'diabetes_model_for_web.h5'
joblib.dump(model, filename)