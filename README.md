# 🎭 IMDB Sentiment Analysis

A complete sentiment analysis pipeline using IMDB movie reviews. This project combines lexicon-based (VADER) and machine learning (Logistic Regression + TF-IDF) approaches, visualizes data, and provides a user-friendly web UI using Streamlit.

---

## 📁 Project Structure

sentiment-analysis/
│
├── app.py # Streamlit Web UI
├── model/
│ ├── analyze_sentiment.py # VADER analysis
│ └── train_model.py # TF-IDF + Logistic Regression model
├── app/
│ └── convert_to_csv.py # Converts IMDB folder to CSV
├── data/
│ └── aclImdb/ # Raw IMDB data from Kaggle
├── output/
│ └── IMDB_Dataset.csv # Preprocessed dataset
├── venv/ # Python virtual environment
├── requirements.txt # Dependencies
└── README.md # This file


---

## 🚀 Features

- ✅ Preprocessing and cleaning of IMDB dataset
- ✅ VADER-based sentiment analysis
- ✅ Logistic Regression with TF-IDF vectorization
- ✅ Streamlit Web App for:
  - Live review input
  - Data visualization (pie chart of sentiment distribution)
  - Predictions using trained model
- ✅ Clean modular code structure

---

## 🔧 Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis

Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux


Prepare Dataset
Download aclImdb dataset from Kaggle IMDB Dataset
Place it inside data/ folder

Run the script to convert to CSV:
python app/convert_to_csv.py

Launch the App
streamlit run app.py

🧪 Example Usage
Type any movie review in the text box
Click "Analyze"
See sentiment result + pie chart of sentiment distribution  

🛠️ Tech Stack
Python 🐍
Streamlit 📊
Pandas, Scikit-learn, NLTK, VADER
Matplotlib
Logistic Regression, TF-IDF

🙋‍♂️ Author
M. Rabi Kiran
LinkedIn | GitHub
