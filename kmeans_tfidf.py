import csv
csv.field_size_limit(10**9)

import argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# Text column options
TEXT_COLS = (
    "news_content","content","text","article_text","body","full_text",
    "clean_content","content_text","news","title","tweet"
)

def pick_first(cols, candidates):
    return next((c for c in candidates if c in cols), None)

def load_df(path, text_col):
    # Load and clean data
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    if not text_col:
        text_col = pick_first(df.columns, TEXT_COLS)
        if not text_col:
            raise KeyError(f"No text column found. Tried {TEXT_COLS}.")
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty dataset after cleaning.")
    return df, text_col

def tfidf(texts, max_features=10000):
    # TF-IDF vectorization
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=max_features, norm=None)
    X = v.fit_transform(texts)
    return normalize(X)

def choose_k(X, kmin, kmax):
    # Run KMeans and calculate metrics
    n = X.shape[0]
    rows, best_sil, best_k, best_labels = [], -1, None, None
    print(f"Running MiniBatchKMeans for k={kmin}..{kmax}...")

    for k in range(kmin, kmax + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=2048, max_no_improvement=5)
        labels = km.fit_predict(X)

        # Sample for metrics
        sample_size = min(5000, n)
        idx_sample = np.random.choice(n, sample_size, replace=False)
        sil = silhouette_score(X[idx_sample], labels[idx_sample]) if k > 1 else np.nan

        try:
            dbi = davies_bouldin_score(X[idx_sample].toarray(), labels[idx_sample]) if k > 1 else np.nan
        except:
            dbi = np.nan

        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil), "davies_bouldin": float(dbi)})

        if k > 1 and sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels

    if best_labels is None:
        km = MiniBatchKMeans(n_clusters=2, random_state=42).fit(X)
        best_k, best_labels = 2, km.labels_

    return pd.DataFrame(rows), best_k, best_labels

def save_plots(metrics, X, labels, outdir):
    # Save plots
    outdir.mkdir(parents=True, exist_ok=True)

    # Elbow
    plt.figure(figsize=(6,4))
    plt.plot(metrics["k"], metrics["inertia"], marker="o")
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(outdir / "elbow.png", dpi=150)
    plt.close()

    # Silhouette
    plt.figure(figsize=(6,4))
    plt.plot(metrics["k"], metrics["silhouette"], marker="o", color="orange")
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(outdir / "silhouette.png", dpi=150)
    plt.close()

    # PCA 2D
    Xv = X.toarray() if X.shape[0] <= 3000 else X[:3000].toarray()
    lv = labels[:Xv.shape[0]]
    xy = PCA(n_components=2, random_state=42).fit_transform(Xv)
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(xy[:,0], xy[:,1], c=lv, s=10, cmap="viridis")
    plt.title("Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.tight_layout()
    plt.savefig(outdir / "pca.png", dpi=150)
    plt.close()

def main():
    # Main process
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="processed_dataset.csv")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--max-features", type=int, default=10000)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    df, text_col = load_df(Path(args.input), args.text_col)
    print(f"Loaded {len(df)} rows; text column = '{text_col}'")

    print("Vectorizing text...")
    X = tfidf(df[text_col].tolist(), max_features=args.max_features)

    metrics, best_k, labels = choose_k(X, args.k_min, args.k_max)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out.insert(0, "cluster", labels)
    out.to_csv(outdir / "clusters.csv", index=False)
    metrics.to_csv(outdir / "metrics.csv", index=False)

    save_plots(metrics, X, labels, outdir)

    print(f"Best k: {best_k}")
    print(f"Results saved in: {outdir}")

if __name__ == "__main__":
    main()
