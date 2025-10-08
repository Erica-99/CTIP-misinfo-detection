# Simple python script to train our best model on its own and save it.
# From the model_training.ipynb notebook it was found that TF-IDF extraction and a Linear SVM model performs best.

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords as nltkstopwords
import joblib

RANDOM_STATE = 17

# Read in dataset
dataset = pd.read_csv("data/processed_dataset.csv", index_col=0)
print("Dataset loaded.")

# Define stopwords
stopwords = set(nltkstopwords.words('english'))
stopwords.update(['said', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'wouldn'])
STOPWORDS = list(stopwords)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, stop_words=STOPWORDS)),
    ('linear_svm', LinearSVC(penalty='l2', max_iter=500))
])

X = dataset['text']
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

pipeline.fit(X_train, y_train)
print("Finished training. Saving model...")

joblib.dump(pipeline, "saved_classifier_model.joblib")
print("Model saved.")

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='real')
recall = recall_score(y_test, y_pred, pos_label='real')
f1 = f1_score(y_test, y_pred, pos_label='real')

print(f"===== Evaluation Results =====")

print(f"Accuracy: {accuracy:0.3f}")
print(f"Precision: {precision:0.3f}")
print(f"Recall: {recall:0.3}")
print(f"F1 Score: {f1:0.3f}")