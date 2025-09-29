# train_compare.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# CONFIG — pick which pair to run
# -----------------------
DATA_DIR = Path(".")
# Examples:
# PolitiFact
#PAIR = ("PolitiFact_fake_news_content.csv", "PolitiFact_real_news_content.csv")

#  BuzzFeed
PAIR = ("BuzzFeed_fake_news_content.csv", "BuzzFeed_real_news_content.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_RESULTS_CSV = "results_comparison.csv"   # set to None to skip saving

# Try these text column names, stop at first match:
TEXT_COL_CANDIDATES = (
    "news_content","content","text","article_text","body",
    "full_text","clean_content","content_text","news","title","tweet"
)

# -----------------------
# Loading
# -----------------------
def load_fake_real_pair(fake_csv: str, real_csv: str):
    fake_df = pd.read_csv(DATA_DIR / fake_csv, encoding="utf-8", engine="python")
    real_df = pd.read_csv(DATA_DIR / real_csv, encoding="utf-8", engine="python")
    fake_df["label"] = "fake"
    real_df["label"] = "real"
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)

    # find a text column
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    if text_col is None:
        raise KeyError(f"No text column found. Tried {TEXT_COL_CANDIDATES}. "
                       f"Available: {df.columns.tolist()}")

    # clean
    df = df.dropna(subset=[text_col, "label"]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    return df, text_col

# -----------------------
# Feature extractors
# -----------------------
def build_vectorizers():
    return {
        "BoW_uni": CountVectorizer(stop_words="english", max_features=20000),
        "TFIDF_uni": TfidfVectorizer(stop_words="english", max_features=20000),
        "TFIDF_uni+bi": TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=30000),
    }

# -----------------------
# Models
# -----------------------
def build_models():
    return {
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=-1), # ← Logistic Regression
        "NaiveBayes": MultinomialNB(),                          # ← Naive Bayes
        "LinearSVM": LinearSVC(),                               # ← Support Vector Machine
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1) # ← Random Forest
    }

# -----------------------
# Training & evaluation
# -----------------------
def evaluate_one(vect_name, vectorizer, model_name, model, X_train, y_train, X_test, y_test):
    # vectorize
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # RandomForest needs dense; others use sparse fine
    if model_name == "RandomForest":
        Xtr_use = Xtr.toarray()
        Xte_use = Xte.toarray()
    else:
        Xtr_use = Xtr
        Xte_use = Xte

    # fit
    model.fit(Xtr_use, y_train)
    pred = model.predict(Xte_use)

    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return {
        "features": vect_name,
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4)
    }

def run_experiments(df, text_col, label_col="label"):
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizers = build_vectorizers()
    models = build_models()

    rows = []
    for vname, vect in vectorizers.items():
        for mname, mdl in models.items():
            print(f"→ {vname} + {mname}")
            res = evaluate_one(vname, vect, mname, mdl, X_train, y_train, X_test, y_test)
            rows.append(res)

    results = pd.DataFrame(rows).sort_values(
        by=["macro_f1","accuracy"], ascending=False
    ).reset_index(drop=True)
    return results

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    fake_csv, real_csv = PAIR
    print(f"Running on: {fake_csv} + {real_csv}")

    df, text_col = load_fake_real_pair(fake_csv, real_csv)
    print("Detected text column:", text_col)
    print("Label counts:\n", df["label"].value_counts(), "\n")

    results = run_experiments(df, text_col)
    print("\n=== RESULTS (sorted by macro F1) ===")
    print(results.to_string(index=False))

    if SAVE_RESULTS_CSV:
        results.to_csv(SAVE_RESULTS_CSV, index=False)
        print(f"\nSaved results to {SAVE_RESULTS_CSV}")
