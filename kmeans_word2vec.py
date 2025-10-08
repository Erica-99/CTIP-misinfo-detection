# Sahan's original script for K-means clustering and the source of the K vs Silhouette score graph.
# Simplified and adapted for consistency by Erica.

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
import umap
plt.switch_backend("Agg")

def train_word2vec(sentences, size=128):
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
    
    print("Running dimension reduction down to 16 dimensions")
    X_16d = umap.UMAP(n_components=16, n_neighbors=15, min_dist=0.1, n_jobs=-1, metric='euclidean').fit_transform(X)
    X_16d = np.nan_to_num(X_16d)
    
    rows, best_sil, best_k, best_labels = [], -1, None, None
    print(f"Running K-Means for k={kmin}..{kmax} using Word2Vec vectors...")

    for k in range(kmin, kmax + 1):
        print(f"Testing k={k}")
        km = MiniBatchKMeans(n_clusters=k, random_state=17, batch_size=512)
        labels = km.fit_predict(X_16d)
        sil = silhouette_score(X_16d, labels)
        try:
            dbi = davies_bouldin_score(X_16d, labels)

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
    plt.title("Silhouette Score vs K-value")
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
    ap.add_argument("--input", default="data/processed_dataset.csv")
    ap.add_argument("--outdir", default="kmeans-out")
    ap.add_argument("--vector-size", type=int, default=128)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv("data/processed_dataset.csv", encoding='utf-8', index_col=0)
    print(f"Loaded {len(df)} rows.")

    print("Preparing sentences for Word2Vec...")
    sentences = [simple_preprocess(t) for t in df['text']]


    print("Training Word2Vec model...")
    model = train_word2vec(sentences, size=args.vector_size)

    print("Vectorizing documents...")
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
