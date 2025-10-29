# preprocess.py
import pandas as pd
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load dataset (tab-separated file inside spam.csv folder)
DATA_PATH = os.path.join('..', 'data', 'spam.csv', 'SMSSpamCollection')
df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])

# 2. Convert labels to binary (spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 3. Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['message'] = df['message'].apply(clean_text)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 5. Vectorize text (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("✅ Data preprocessing completed.")
