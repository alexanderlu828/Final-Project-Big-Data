import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Step 1: Load the public dataset ===
df = pd.read_csv("public_data.csv")  # Replace with your actual path
X = df.drop(columns=["id"])  # Drop the ID column

# === Step 2: Standardize the data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: Fit Gaussian Mixture Model ===
n_clusters = 15  # For 4D data: 4n - 1 = 15, 6D data: 4n - 1 = 23

gmm = GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# === Save the predictions for submission ===
submission = pd.DataFrame({
    "id": df["id"],
    "label": gmm_labels
})
submission.to_csv("public_submission.csv", index=False)

# === Step 4: Visualize clusters using PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="tab20", s=30)
plt.title("GMM Cluster labels (15) visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster Label")
plt.grid(True)
plt.tight_layout()
# === Save the visualization ===
plt.savefig("public_results.png", dpi=300)

# Private dataset

# === Step 1: Load the public dataset ===
df = pd.read_csv("private_data.csv")  # Replace with your actual path
X = df.drop(columns=["id"])  # Drop the ID column

# === Step 2: Standardize the data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: Fit Gaussian Mixture Model ===
n_clusters = 23  # For 4D data: 4n - 1 = 15, 6D data: 4n - 1 = 23

gmm = GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# === Save the predictions for submission ===
submission = pd.DataFrame({
    "id": df["id"],
    "label": gmm_labels
})
submission.to_csv("private_submission.csv", index=False)

# === Step 4: Visualize clusters using PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="tab20", s=30)
plt.title("GMM Cluster labels (23) visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster Label")
plt.grid(True)
plt.tight_layout()
# === Save the visualization ===
plt.savefig("private_results.png", dpi=300)

