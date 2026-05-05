import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)
print("Dataset shape:", df.shape)
print(df["label"].value_counts())


# ─────────────────────────────────────────────
# 2. CLEAN TEXT
# ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()                              # lowercase
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()         # remove extra spaces
    return text

df["message"] = df["message"].apply(clean_text)


# ─────────────────────────────────────────────
# 3. TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.2, random_state=42
)


# ─────────────────────────────────────────────
# 4. TF-IDF FEATURES (improved with n-grams)
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams (e.g. "free money", "click here")
    max_df=0.95,          # ignore words appearing in >95% of messages
    min_df=2              # ignore words appearing in only 1 message
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print("Feature shape:", X_train_vec.shape)


# ─────────────────────────────────────────────
# 5. TRAIN NAIVE BAYES
# ─────────────────────────────────────────────
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)


# ─────────────────────────────────────────────
# 6. TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

# Custom threshold — lower value = more aggressive spam catching
CUSTOM_THRESHOLD = 0.3
y_prob_lr = lr_model.predict_proba(X_test_vec)[:, 1]
y_pred_lr_custom = ["spam" if p >= CUSTOM_THRESHOLD else "ham" for p in y_prob_lr]


# ─────────────────────────────────────────────
# 7. EVALUATION — print reports
# ─────────────────────────────────────────────
for name, preds in [("Naive Bayes", y_pred_nb),
                    ("Logistic Regression (default)", y_pred_lr),
                    (f"Logistic Regression (threshold={CUSTOM_THRESHOLD})", y_pred_lr_custom)]:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))


# ─────────────────────────────────────────────
# 8. MODEL COMPARISON TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame({
    "Model": ["Naive Bayes", "Logistic Regression"],
    "Accuracy":  [accuracy_score(y_test, y_pred_nb),
                  accuracy_score(y_test, y_pred_lr)],
    "Precision": [precision_score(y_test, y_pred_nb, pos_label="spam"),
                  precision_score(y_test, y_pred_lr, pos_label="spam")],
    "Recall":    [recall_score(y_test, y_pred_nb,    pos_label="spam"),
                  recall_score(y_test, y_pred_lr,    pos_label="spam")],
    "F1-Score":  [f1_score(y_test, y_pred_nb,        pos_label="spam"),
                  f1_score(y_test, y_pred_lr,         pos_label="spam")]
})
print("\n\nModel Comparison:\n")
print(results.to_string(index=False))


# ─────────────────────────────────────────────
# 9. VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, preds, title in zip(axes,
                             [y_pred_nb, y_pred_lr],
                             ["Naive Bayes", "Logistic Regression"]):
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["ham", "spam"],
                yticklabels=["ham", "spam"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

# Bar chart comparing metrics
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
nb_scores = results[results["Model"] == "Naive Bayes"][metrics].values[0]
lr_scores = results[results["Model"] == "Logistic Regression"][metrics].values[0]

x = range(len(metrics))
width = 0.35
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.bar([i - width/2 for i in x], nb_scores, width, label="Naive Bayes",       color="steelblue")
ax2.bar([i + width/2 for i in x], lr_scores, width, label="Logistic Regression", color="coral")
ax2.set_xticks(list(x))
ax2.set_xticklabels(metrics)
ax2.set_ylim(0.7, 1.02)
ax2.set_ylabel("Score")
ax2.set_title("Model Performance Comparison")
ax2.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


# ─────────────────────────────────────────────
# 10. USER INPUT INTERFACE
# ─────────────────────────────────────────────
def predict_message(message, model, vectorizer, threshold=0.5):
    cleaned    = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prob       = model.predict_proba(vectorized)[0][1]   # P(spam)
    label      = "SPAM" if prob >= threshold else "HAM"
    print(f"\n  Message  : {message}")
    print(f"  Result   : {label}")
    print(f"  Spam prob: {prob:.2%}")
    return label

print("\n\n─── Spam Detector (type 'quit' to exit) ───")
while True:
    user_input = input("\nEnter a message: ").strip()
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    predict_message(user_input, lr_model, vectorizer, threshold=CUSTOM_THRESHOLD)