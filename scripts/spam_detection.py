import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve
)

# Basic Stopword List
stop_words = set([
    "the", "a", "an", "is", "it", "to", "and", "in", "this", "of", "on", "for", "with", "at", "by", "from"
])

# Load Dataset 
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df["clean_text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["label_num"], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


models = {
    "Logistic Regression": (LogisticRegression(solver="liblinear"), {"C": [0.1, 1, 10]}),
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}),
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    })
}

# Train and Evaluate
results = {}
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    search = GridSearchCV(model, params, cv=5) if name != "Random Forest" else RandomizedSearchCV(model, params, n_iter=5, cv=5)
    search.fit(X_train_vec, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_vec)
    y_prob = best_model.predict_proba(X_test_vec)[:, 1]

    results[name] = {
        "model": best_model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("Best Params:", search.best_params_)
    print("Accuracy:", results[name]["accuracy"])
    print("Precision:", results[name]["precision"])
    print("Recall:", results[name]["recall"])
    print("F1 Score:", results[name]["f1"])
    print("ROC AUC:", results[name]["roc_auc"])
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot ROC Curve 
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
