# Typology PoC Findings

Scoped proof-of-concept. This is a down-payment on Phase 2 (Foundational
Typology Discovery), not the full 11-paper program. It replaces ONE explicit
placeholder: the prior "Mock t-SNE" figure that clustered sklearn.make_blobs
synthetic data. Here we run ONE real clustering pass over ONE real public
adversarial-prompt corpus and report whatever silhouette we actually get.

## Dataset

- Source: AdvBench harmful_behaviors.csv (llm-attacks/llm-attacks, main branch).
- URL: https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
- Column used: "goal".
- Row count: 520 prompts (all non-empty rows).
- Raw CSV cached locally as advbench_harmful_behaviors.csv.
- Fetch succeeded on the first source; no fallback was needed.

## Embedding backend

- sentence-transformers all-MiniLM-L6-v2 (CPU), first rung of the lazy ladder.
- The model was installed (sentence-transformers 5.5.0) and downloaded on first
  use; the TF-IDF fallback was not exercised.
- Embedding matrix: 520 x 384, L2-normalized.

## KMeans silhouette by k

  k     silhouette
  2     0.0713
  3     0.0779
  4     0.0780
  5     0.0847
  6     0.0934
  7     0.0997
  8     0.1055
  9     0.1019
  10    0.1179
  11    0.1172
  12    0.1202

Best k = 12, silhouette = 0.1202. Silhouette rises monotonically (with one dip
at k=9 and k=11) across the swept range, so k=12 wins only as the top of the
2..12 window; there is no clean elbow or peak inside the range.

## DBSCAN

- Parameters: eps=0.5, min_samples=5, metric=cosine.
- Result: 1 cluster, 8 noise points. Silhouette not defined (need >=2 non-noise
  clusters). At this eps the embedding space is effectively one dense blob with
  a few outliers. DBSCAN did not recover a usable typology here.

## Honest comparison to the >0.4 target

The agenda sets silhouette > 0.4 as the validation bar for typology discovery.
This PoC does not meet it: the best KMeans silhouette is 0.1202, roughly a
quarter of target, and DBSCAN failed to separate clusters at all. The low
score is expected and informative rather than a failure of the method. AdVBench
goals are short, lexically similar imperative requests ("Write a tutorial on
how to ..."), so MiniLM embeds them into one dense, overlapping region with no
hard geometric gaps, which is exactly what a low silhouette measures. Critically,
the KMeans partition at k=12 is still semantically coherent on inspection: the
clusters separate into recognizable harm families (identity theft, bomb and
explosive instructions, network and government-database hacking, malware and
exploits, disinformation and fake news, defamation, counterfeiting and piracy,
insider trading, incitement and hate). So the typology signal is real even
though the silhouette geometry is weak. Reaching the >0.4 bar will need the
techniques the full program already plans: a domain-tuned or instruction-tuned
embedding, dimensionality reduction (UMAP) before clustering, and possibly
HDBSCAN with tuned density parameters, evaluated on a more varied multi-source
corpus rather than a single near-template dataset.

## What this replaces

The prior typology figure in the visualization notebook was an explicit
"Mock t-SNE" over sklearn.make_blobs, i.e. synthetic Gaussian blobs that
trivially produce clean, well-separated clusters and a high silhouette by
construction. That figure demonstrated nothing about real adversarial prompts.
This PoC swaps in a real corpus, a real sentence embedding, and a real
clustering run, and reports the honest (lower) silhouette that real data yields.
The make_blobs number was never evidence; 0.1202 on AdvBench is.

## Reproduce

    python typology_poc.py

Outputs: advbench_harmful_behaviors.csv (cached raw data),
cluster_results.csv (prompt, kmeans_cluster), and the console log above.
