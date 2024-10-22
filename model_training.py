import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the CSV files into pandas DataFrames
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add a label column to each DataFrame: 1 for True news, 0 for Fake news
true_df['label'] = 1
fake_df['label'] = 0

# Combine the DataFrames
news_df = pd.concat([true_df, fake_df])

# Drop rows with null values in the 'text' column
news_df = news_df.dropna(subset=['text'])

# Split the data into features and labels
X = news_df['text']
y = news_df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to feature vectors
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
lr = LogisticRegression()

# Train the model
lr.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = lr.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")

import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define a function to make predictions using the loaded model
def predict(text):
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return "True" if prediction[0] == 1 else "Fake"

# Example usage
example_text = "Example news text to classify"
prediction = predict(example_text)
print(f"Prediction: {prediction}")
