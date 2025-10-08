import csv
csv.field_size_limit(10**9)

import argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# Detect text column automatically
TEXT_COLS = (
    "news_content","content","text","article_text","body","full_text",
    "clean_content","content_text","news","title","tweet"
)

def pick_first(cols, candidates):
    return next((c for c in candidates if c in cols), None)

def load_df(path, text_col):
    # Load and clean dataset
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    if not text_col:
        text_col = pick_first(df.columns, TEXT_COLS)
        if not text_col:
            raise KeyError(f"No text column found. Tried {TEXT_COLS}.")
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty after removing blanks.")
    return df, text_col

def train_word2vec(sentences, size=100):
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=size, window=5, min_count=2, workers=4, sg=1)
    return model

def vectorize_texts(model, sentences):
    # Average word vectors for each text
    vectors = []
    for words in sentences:
        word_vecs = [model.wv[w] for w in words if w in model.wv]
        if len(word_vecs) > 0:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def choose_k(X, kmin, kmax):
    # Run MiniBatchKMeans for different k values
    rows, best_sil, best_k, best_labels = [], -1, None, None
    print(f"Running K-Means for k={kmin}..{kmax} using Word2Vec vectors...")

    for k in range(kmin, kmax + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        try:
            dbi = davies_bouldin_score(X, labels)
        except:
            dbi = np.nan
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": dbi})
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels

    return pd.DataFrame(rows), best_k, best_labels

def save_plots(metrics, X, labels, outdir):
    # Save silhouette and PCA plots
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(metrics["k"], metrics["silhouette"], marker="o", color="orange")
    plt.title("Silhouette Score vs K (Word2Vec)")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(outdir / "silhouette_word2vec.png", dpi=150)
    plt.close()

    Xv = X if X.shape[0] <= 3000 else X[:3000]
    lv = labels[:Xv.shape[0]]
    xy = PCA(n_components=2, random_state=42).fit_transform(Xv)
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(xy[:,0], xy[:,1], c=lv, s=10, cmap="viridis")
    plt.title("Clusters (Word2Vec + PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.tight_layout()
    plt.savefig(outdir / "pca_word2vec.png", dpi=150)
    plt.close()

def main():
    # Main process
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="processed_dataset.csv")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--vector-size", type=int, default=100)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    args = ap.parse_args()

    df, text_col = load_df(Path(args.input), args.text_col)
    print(f"Loaded {len(df)} rows; text column = '{text_col}'")

    print("Preparing sentences for Word2Vec...")
    sentences = [simple_preprocess(t) for t in df[text_col]]

    print("Training Word2Vec model...")
    model = train_word2vec(sentences, size=args.vector_size)

    print("Vectorizing sentences...")
    X = vectorize_texts(model, sentences)

    metrics, best_k, labels = choose_k(X, args.k_min, args.k_max)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out.insert(0, "cluster", labels)
    out.to_csv(outdir / "clusters_word2vec.csv", index=False)
    metrics.to_csv(outdir / "metrics_word2vec.csv", index=False)

    save_plots(metrics, X, labels, outdir)

    print(f"Best k: {best_k}")
    print(f"Results saved in: {outdir}")

if __name__ == "__main__":
    main()
