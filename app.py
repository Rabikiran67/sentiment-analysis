import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# Load model and vectorizer
model = load('output/sentiment_model.joblib')
vectorizer = load('output/tfidf_vectorizer.joblib')

# UI
st.title("🎬 IMDB Sentiment Analyzer")
page = st.sidebar.radio("Navigation", ["📊 Visualize Data", "📝 User Input"])

df = pd.read_csv("output/IMDB_Dataset.csv")

if page == "📊 Visualize Data":
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    st.dataframe(df.sample(5))

elif page == "📝 User Input":
    st.subheader("Type a movie review")
    user_input = st.text_area("Enter text here:")
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Enter a review first.")
        else:
            X = vectorizer.transform([user_input])
            result = model.predict(X)[0]
            st.success(f"Predicted Sentiment: **{result.capitalize()}**")
