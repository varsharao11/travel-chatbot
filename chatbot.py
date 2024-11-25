import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import re

# Step 1: Data Collection (Assume you have a CSV file 'disease_symptoms.csv')
data = pd.read_csv('disease_symptoms.csv')
# Preprocess your data, handle missing values, etc.

# Step 2: Feature Extraction
# Vectorize symptoms if they are in text form
vectorizer = CountVectorizer()

# Step 3: Split Data
X = vectorizer.fit_transform(data['Symptoms'])
y = data['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# Step 6: Chatbot Logic
def chatbot():
    print("Hello! I am your health assistant. Please tell me your symptoms.")
    user_input = input()
    processed_input = vectorizer.transform([user_input])
    prediction = model.predict(processed_input)
    print(f"You may have: {prediction[0]}")


chatbot()

# This is a very basic example, and it would need to be expanded significantly.
