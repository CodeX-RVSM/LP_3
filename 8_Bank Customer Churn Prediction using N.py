# Bank Customer Churn Prediction using Neural Networks
# Author: Rushikesh Mangalkar

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('Churn_Modelling.csv')

# Data info
df.info()

# Visualize churn distribution
plt.xlabel('Exited')
plt.ylabel('Count')
df['Exited'].value_counts().plot.bar()
plt.show()

# Check unique values of Geography
print(df['Geography'].value_counts())

# One-hot encoding for Geography and Gender
df = pd.concat([df, pd.get_dummies(df['Geography'], prefix='Geo')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Gender'])], axis=1)

# Drop unnecessary columns
df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender'], inplace=True)

# Prepare data for model
y = df['Exited'].values
x = df.loc[:, df.columns != 'Exited'].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, test_size=0.25)

# Standardize data
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)

# TensorFlow Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(x_train.shape[1],)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='BinaryCrossentropy')

# Train the model
model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=100)

# Predictions
pred = model.predict(x_test)

# Threshold predictions at 0.5
y_pred = [1 if val > 0.5 else 0 for val in pred]

# Evaluate TensorFlow model
print("TensorFlow NN Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(cm)
display.plot()
plt.show()

# Scikit-learn MLP Classifier
nn_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=300)
nn_classifier.fit(x_train, y_train)

# Predictions
y_pred2 = nn_classifier.predict(x_test)

# Accuracy Scores
print("Sklearn MLPClassifier Accuracy:", accuracy_score(y_test, y_pred2))
print("Model Score:", nn_classifier.score(x_test, y_test))
