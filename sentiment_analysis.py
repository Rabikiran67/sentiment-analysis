import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from joblib import dump
import nltk

nltk.download('stopwords')

# Load dataset
df = pd.read_csv('output/IMDB_Dataset.csv')

# VADER sentiment labeling
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment'] = df['review'].apply(get_sentiment)

# ML model training
X = df['review']
y = df['Sentiment']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, preds))

# Save model and vectorizer
dump(model, 'output/sentiment_model.joblib')
dump(vectorizer, 'output/tfidf_vectorizer.joblib')
print("✅ Model and vectorizer saved.")
