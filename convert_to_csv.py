import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def read_reviews(folder_path, sentiment_label):
    reviews = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), encoding='utf-8') as file:
            content = file.read()
            reviews.append((content, sentiment_label))
    return reviews

def create_csv():
    dataset = []
    base_path = 'data/aclImdb'
    folders = [
        ('train/pos', 'positive'),
        ('train/neg', 'negative'),
        ('test/pos', 'positive'),
        ('test/neg', 'negative')
    ]
    
    for subfolder, label in folders:
        path = os.path.join(base_path, subfolder)
        dataset.extend(read_reviews(path, label))
    
    df = pd.DataFrame(dataset, columns=['review', 'sentiment'])

    # Add VADER Sentiment
    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        score = analyzer.polarity_scores(text)
        if score['compound'] >= 0.05:
            return 'positive'
        elif score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['Sentiment'] = df['review'].apply(get_vader_sentiment)

    # Save to output
    output_path = 'output/IMDB_Dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ CSV created at {output_path}")
    print(df.head())

if __name__ == "__main__":
    create_csv()
