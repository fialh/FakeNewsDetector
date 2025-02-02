import joblib
import re
import string

# Load Model and Vectorizer
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    print("Model or vectorizer file not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the model or vectorizer: {e}")
    exit(1)

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Prediction Function
def predict_news(text):
    text_clean = clean_text(text)
    text_tfidf = vectorizer.transform([text_clean])
    prediction = model.predict(text_tfidf)[0]
    return "FAKE" if prediction == 1 else "REAL"

# Example Usage
news_article = input("Enter a news article: ")
print(f"Prediction: {predict_news(news_article)}")