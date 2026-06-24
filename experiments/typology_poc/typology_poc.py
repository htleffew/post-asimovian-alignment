"""
Typology PoC: ONE real clustering run on ONE real public adversarial-prompt
dataset. Scoped down-payment replacing the prior "Mock t-SNE over make_blobs"
synthetic figure in the Post-Asimovian Framework agenda (Phase 2: Foundational
Typology Discovery). NOT the full 11-paper program.

Pipeline: fetch AdvBench harmful_behaviors.csv (cache locally) -> embed via a
lazy ladder (sentence-transformers MiniLM, else TF-IDF) -> KMeans k=2..12 with
silhouette per k -> pick best k -> DBSCAN -> write cluster_results.csv +
per-cluster sample prompts.
"""
import csv
import io
import os
import sys
import urllib.request

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_CSV = os.path.join(HERE, "advbench_harmful_behaviors.csv")
RESULTS_CSV = os.path.join(HERE, "cluster_results.csv")

DATA_SOURCES = [
    ("AdvBench harmful_behaviors",
     "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
     "goal"),
    ("HarmBench behaviors",
     "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
     "Behavior"),
]


def fetch_dataset():
    """Return (source_name, list_of_prompts). Cache raw CSV locally."""
    if os.path.exists(CACHE_CSV):
        with open(CACHE_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        col = rows[0].keys()
        textcol = "goal" if "goal" in col else ("Behavior" if "Behavior" in col else list(col)[0])
        prompts = [r[textcol].strip() for r in rows if r.get(textcol, "").strip()]
        print("Loaded %d prompts from local cache %s" % (len(prompts), CACHE_CSV))
        return ("AdvBench harmful_behaviors (cached)", prompts)

    last_err = None
    for name, url, textcol in DATA_SOURCES:
        try:
            print("Fetching %s from %s" % (name, url))
            req = urllib.request.Request(url, headers={"User-Agent": "typology-poc/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            rows = list(csv.DictReader(io.StringIO(raw)))
            if not rows:
                raise ValueError("empty CSV")
            if textcol not in rows[0]:
                textcol = list(rows[0].keys())[0]
            prompts = [r[textcol].strip() for r in rows if r.get(textcol, "").strip()]
            if len(prompts) < 20:
                raise ValueError("too few rows: %d" % len(prompts))
            with open(CACHE_CSV, "w", encoding="utf-8", newline="") as f:
                f.write(raw)
            print("Fetched %d prompts. Cached to %s" % (len(prompts), CACHE_CSV))
            return (name, prompts)
        except Exception as e:
            print("  FAILED: %s" % e)
            last_err = e
    raise RuntimeError("All network fetches failed. Last error: %s" % last_err)


def embed(prompts):
    """Lazy ladder: MiniLM if available, else TF-IDF. Return (backend, matrix)."""
    try:
        from sentence_transformers import SentenceTransformer
        print("Embedding backend: sentence-transformers all-MiniLM-L6-v2")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode(prompts, show_progress_bar=False, normalize_embeddings=True)
        return ("sentence-transformers all-MiniLM-L6-v2", np.asarray(X))
    except Exception as e:
        print("  MiniLM unavailable (%s). Falling back to TF-IDF." % e)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=4096, stop_words="english")
        X = vec.fit_transform(prompts).toarray()
        return ("sklearn TfidfVectorizer", X)


def main():
    source_name, prompts = fetch_dataset()
    backend, X = embed(prompts)
    print("Embedded matrix shape: %s" % (X.shape,))

    print("\nKMeans silhouette by k:")
    sil_by_k = {}
    best_k, best_sil, best_labels = None, -1.0, None
    for k in range(2, 13):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_by_k[k] = sil
        print("  k=%2d  silhouette=%.4f" % (k, sil))
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels
    print("\nBest k=%d  silhouette=%.4f" % (best_k, best_sil))

    # DBSCAN. eps chosen from median pairwise scale; report whatever it gives.
    print("\nDBSCAN:")
    db = DBSCAN(eps=0.5, min_samples=5, metric="cosine")
    db_labels = db.fit_predict(X)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int(np.sum(db_labels == -1))
    db_sil = "n/a (need >=2 non-noise clusters)"
    mask = db_labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        try:
            db_sil = "%.4f" % silhouette_score(X[mask], db_labels[mask], metric="cosine")
        except Exception as e:
            db_sil = "n/a (%s)" % e
    print("  eps=0.5 min_samples=5 metric=cosine -> %d clusters, %d noise points, silhouette=%s"
          % (n_clusters, n_noise, db_sil))

    # Write cluster_results.csv
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "kmeans_cluster"])
        for p, c in zip(prompts, best_labels):
            w.writerow([p, int(c)])
    print("\nWrote %s (%d rows)" % (RESULTS_CSV, len(prompts)))

    # Per-cluster sample of 3 prompts for human inspection
    print("\nPer-cluster samples (best k=%d):" % best_k)
    for c in range(best_k):
        idx = [i for i, lab in enumerate(best_labels) if lab == c]
        print("\n  Cluster %d (n=%d):" % (c, len(idx)))
        for i in idx[:3]:
            print("    - %s" % prompts[i][:100])

    # Return a compact summary line for the findings doc
    print("\nSUMMARY backend=%s source=%s n=%d best_k=%d best_sil=%.4f db_clusters=%d db_noise=%d"
          % (backend, source_name, len(prompts), best_k, best_sil, n_clusters, n_noise))


if __name__ == "__main__":
    main()
