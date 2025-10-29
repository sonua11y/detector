# model_train.py
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocess import X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# 1. Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 2. Predictions
y_pred = model.predict(X_test_tfidf)

# 3. Evaluation
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Save model and vectorizer
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))
os.makedirs(app_dir, exist_ok=True)
joblib.dump((model, vectorizer), os.path.join(app_dir, 'model.pkl'))

print("✅ Model trained and saved successfully.")
