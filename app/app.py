import streamlit as st
import pickle
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Resolve important paths
app_dir = os.path.dirname(__file__)
model_path = os.path.join(app_dir, "model.pkl")
data_path = os.path.abspath(os.path.join(app_dir, "..", "data", "spam.csv", "SMSSpamCollection"))

# Bump this when changing model/vectorizer behavior
MODEL_VERSION = 3


def ensure_nltk(resource: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)


def clean_text_factory() -> tuple:
    ensure_nltk('corpora/stopwords')
    # WordNet lemmatizer sometimes needs omw-1.4
    ensure_nltk('corpora/wordnet')
    ensure_nltk('corpora/omw-1.4')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def safe_lemmatize(token: str) -> str:
        try:
            return lemmatizer.lemmatize(token)
        except Exception:
            return token

    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [safe_lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)

    return clean_text, lemmatizer, stop_words


def train_and_save_model() -> tuple:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Ensure the repository structure is intact.")

    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    clean_text, _, _ = clean_text_factory()
    df['cleaned'] = df['message'].apply(clean_text)

    # Use word unigrams + bigrams and sublinear TF to better capture promotional phrases
    # Lower min_df to catch rare promotional phrases, increase features for better coverage
    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95
    )
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label'].map({'ham': 0, 'spam': 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LinearSVC is a strong baseline for sparse text; balance classes
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # Persist along with a simple version stamp
    with open(model_path, 'wb') as f:
        pickle.dump({
            'version': MODEL_VERSION,
            'model': model,
            'tfidf': tfidf,
        }, f)

    return model, tfidf


def load_or_build_model() -> tuple:
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
                if isinstance(payload, dict) and payload.get('version') == MODEL_VERSION:
                    return payload['model'], payload['tfidf']
        except Exception:
            # Corrupted/old; rebuild below
            pass
    # Build if missing or failed to load
    st.info("Preparing model... (first run may take ~30s)")
    return train_and_save_model()


model, tfidf = load_or_build_model()
clean_text, _, _ = clean_text_factory()

st.title("📧 Spam Email Detector")
st.write("This app detects whether a given message or email is Spam or Not Spam.")

# User input
input_text = st.text_area("Enter your message or email text here:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Clean and transform input text (same preprocessing as training)
        cleaned_input = clean_text(input_text)
        transformed = tfidf.transform([cleaned_input])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🚨 This looks like SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
