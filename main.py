import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD DATA
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# 2. CLEAN TEXT
def clean_text(text):
    text = text.lower()                         # lowercase
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text

df["message"] = df["message"].apply(clean_text)

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# 4. TF-IDF FEATURES
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Feature shape:", X_train_vec.shape)

# 5. TRAIN MODEL (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. PREDICT
y_pred = model.predict(X_test_vec)


# 7. EVALUATION
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))