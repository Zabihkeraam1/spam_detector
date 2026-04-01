import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

DATA_PATH = "data/processed/spam_cleaned.csv"
data = pd.read_csv(DATA_PATH)

data = data.dropna(subset=['message', 'label'])
data['message'] = data['message'].astype(str)
data['label'] = data['label'].astype(int)
data = data[data['message'].str.strip() != ""]

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "NaiveBayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.1, 0.5, 1.0]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "params": {
            "C": [0.1, 1, 10]
        }
    },
    "SVM": {
        "model": LinearSVC(),
        "params": {
            "C": [0.1, 1, 10]
        }
    }
}

results = []
best_model = None
best_score = 0
best_name = ""

for name, config in models.items():
    print(f"\n🔍 Training {name}...")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train_tfidf, y_train)

    model = grid.best_estimator_

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(y_test, y_pred))

    results.append((name, acc, f1))

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

print("\n📊 Model Comparison:")
for name, acc, f1 in results:
    print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")

print(f"\n🏆 Best Model: {best_name} (F1: {best_score:.4f})")

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Best model and vectorizer saved!")