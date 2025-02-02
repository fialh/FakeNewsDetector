# Project Collaborators: Uzair Azizuddin, Firas Al Halaq, Christian Pimentel, Awab Elfadl

# Purpose of Project: The goal is to develop a system that can automatically 
# classify news articles as real or fake based on their content. The process involves collecting data,
# preprocessing it, training a machine learning model, and then evaluating the model's performance.

# This file is meant to predict the news using the model and vectorizer that were trained in fake_news_train.py


import joblib
import re
import string
import os

# Print current working directory
print("Current working directory:", os.getcwd())
 #Can get the file path so we can see if the file is in the correct location and we can
# change the file path if needed

# Load Model and Vectorizer
try:
    model_path = "/Users/ENTER_USERNAME/FILE_LOCATION/FakeNewsDetector/fake_news_detector/fake_news_model.pkl" #Change this to your file location
    vectorizer_path = "/Users/ENTER_USERNAME/FILE_LOCATION/FakeNewsDetector/fake_news_detector/tfidf_vectorizer.pkl" #Change this to your file location
    
    if not os.path.exists(model_path): #Check if the file exists. If not, raise an error. 
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    
    # Load the model and vectorizer, and print a message that says that the model and vectorizer were loaded successfully
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")

except FileNotFoundError as fnf_error: #If the file path is incorrect, this will be flagged. If this is flagged, then the file path is incorrect. CHANGE THE FILE PATH.
    print(fnf_error)
    exit(1)
except Exception as e: #If there is any other error, this will be flagged. If this is flagged, then the model and vectorizer were not loaded successfully.
    print(f"An error occurred while loading the model or vectorizer: {e}")
    exit(1)

# Text Preprocessing Function
def clean_text(text): #This function will clean the text by removing punctuation, numbers, and converting to lowercase
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Prediction Function
def predict_news(text): #This function will take the text as input and return whether the news is real or fake
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return "FAKE" if prediction == 1 else "REAL"


# This is a sample text to test the prediction function
if __name__ == "__main__":
    sample_text = "The stock market crashed due to economic uncertainty."
    prediction = predict_news(sample_text)
    print(f"Prediction: {prediction}")
