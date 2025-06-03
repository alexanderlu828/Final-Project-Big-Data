import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load public dataset ===
df = pd.read_csv("public_data.csv")  # Update path if needed
X = df.drop(columns=["id"])

# === Step 1: Preprocessing ===
qt = QuantileTransformer(output_distribution="normal", random_state=42)
X_qt = qt.fit_transform(X)

# === Step 2: Use [S2, S3] for DBSCAN clustering ===
X_proj = X[["2", "3"]].values
dbscan = DBSCAN(eps=300, min_samples=20)
macro_labels = dbscan.fit_predict(X_proj)

# === Step 3: GMM within each DBSCAN macro-cluster ===
final_labels = np.full(len(X), -1, dtype=int)
label_offset = 0

for cluster_id in np.unique(macro_labels):
    if cluster_id == -1:
        continue  # Skip noise

    mask = (macro_labels == cluster_id)
    X_cluster = X_qt[mask]

    # Number of subclusters per macrocluster (manual/tuned/auto)
    est_n_clusters = min(5, len(X_cluster))  # Tune this if needed
    gmm = GaussianMixture(n_components=est_n_clusters, covariance_type="full", random_state=42)
    sub_labels = gmm.fit_predict(X_cluster)

    final_labels[mask] = sub_labels + label_offset
    label_offset += sub_labels.max() + 1

# === Step 4: Save cluster labels ===
submission = pd.DataFrame({
    "id": df["id"],
    "label": final_labels
})
submission.to_csv("public_submission.csv", index=False)

# === Step 5: Visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_qt)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap="tab20", s=30)
plt.title("Hybrid DBSCAN + GMM Clustering (PCA view)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster Label")
plt.grid(True)
plt.tight_layout()
plt.savefig("public_plot.png", dpi=300)


