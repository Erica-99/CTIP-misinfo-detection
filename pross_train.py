# Dinuka's original script for training classifiers. Portions of code were recycled into the main training notebook.
# Simplified and made consistent with the rest of the codebase by Erica.

import csv
csv.field_size_limit(10**9)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from nltk.corpus import stopwords as nltkstopwords
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# CONFIG
# =======================
MAIN_FILE = "data/processed_dataset.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 17
SAVE_RESULTS_CSV = "results_comparison.csv"   # set to None to skip saving

stopwords = set(nltkstopwords.words('english'))
stopwords.update(['said', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'wouldn'])
STOPWORDS = list(stopwords)

# =======================
# Feature extractors
# =======================
def build_vectorizers():
    return {
        "BoW_uni": CountVectorizer(ngram_range=(1,1), min_df=5, stop_words=STOPWORDS),
        "TFIDF_uni": TfidfVectorizer(ngram_range=(1,1), min_df=5, stop_words=STOPWORDS),
        "TFIDF_uni+bi": TfidfVectorizer(ngram_range=(1,2), min_df=5, stop_words=STOPWORDS)
    }

# =======================
# Models
# =======================
def build_models():
    return {
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=-1),  # Logistic Regression
        "NaiveBayes": MultinomialNB(),                           # Multinomial Naive Bayes
        "LinearSVM": LinearSVC(penalty='l2', max_iter=2000),     # Linear SVM
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)  # Random Forest
    }

# =======================
# Training & evaluation
# =======================
def evaluate_one(vect_name, vectorizer, model_name, model, X_train, y_train, X_test, y_test):
    # Vectorize
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # RandomForest needs dense. Others can use sparse
    if model_name == "RandomForest":
        Xtr_use = Xtr.toarray()
        Xte_use = Xte.toarray()
    else:
        Xtr_use = Xtr
        Xte_use = Xte

    # Fit & predict
    model.fit(Xtr_use, y_train)
    pred = model.predict(Xte_use)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, pos_label='real')
    rec = recall_score(y_test, pred, pos_label='real')
    f1 = f1_score(y_test, pred, pos_label='real')
    
    return {
        "features": vect_name,
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4)
    }

def run_experiments(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    vectorizers = build_vectorizers()
    models = build_models()

    rows = []
    for vname, vect in vectorizers.items():
        for mname, mdl in models.items():
            print(f"-> {vname} + {mname}")
            res = evaluate_one(vname, vect, mname, mdl, X_train, y_train, X_test, y_test)
            rows.append(res)

    results = pd.DataFrame(rows).sort_values(by=["f1", "accuracy"], ascending=False).reset_index(drop=True)
    return results

# =======================
# Main
# =======================
if __name__ == "__main__":
    print(f"Running on processed dataset: {MAIN_FILE}")
    df = pd.read_csv(MAIN_FILE, encoding="utf-8", index_col=0)

    results = run_experiments(df)
    
    print("\n=== RESULTS (Sorted by F1) ===")
    print(results.to_string(index=False))

    if SAVE_RESULTS_CSV:
        results.to_csv(SAVE_RESULTS_CSV, index=False)
        print(f"\nSaved results to {SAVE_RESULTS_CSV}")
