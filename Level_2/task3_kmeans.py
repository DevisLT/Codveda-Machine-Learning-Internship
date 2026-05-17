import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("datasets/iris.csv")
le = LabelEncoder()
true_labels = le.fit_transform(df['species'])

X = df.drop(columns=['species'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset: {df.shape}")

K_range   = range(2, 11)
inertias  = []
silhouettes = []

print(f"\n{'K':>4}  {'WCSS':>14}  {'Silhouette':>12}")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"{k:>4}  {km.inertia_:>14.2f}  {silhouette_score(X_scaled, km.labels_):>12.4f}")

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
km_final.fit(X_scaled)
labels = km_final.labels_

ari = adjusted_rand_score(true_labels, labels)
sil = silhouette_score(X_scaled, labels)

print(f"\nFinal K=3 Results:")
print(f"Inertia          : {km_final.inertia_:.4f}")
print(f"Silhouette Score : {sil:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Cluster sizes    : {dict(zip(*np.unique(labels, return_counts=True)))}")

print("\nCluster Centroids (original scale):")
centroids = scaler.inverse_transform(km_final.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=X.columns)
print(centroid_df.round(2))

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cluster_colors = ['#e74c3c', '#2ecc71', '#3498db']
species_colors = ['#f39c12', '#9b59b6', '#1abc9c']

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("K-Means Clustering - Iris", fontsize=14, fontweight='bold')

axes[0].plot(K_range, inertias, 'bo-', markersize=6)
axes[0].axvline(3, color='red', linestyle='--', alpha=0.7, label='K=3')
axes[0].set_xlabel("K")
axes[0].set_ylabel("WCSS (Inertia)")
axes[0].set_title("Elbow Method")
ax0b = axes[0].twinx()
ax0b.plot(K_range, silhouettes, 'r^--', markersize=6, alpha=0.6)
ax0b.set_ylabel("Silhouette Score", color='red')
axes[0].legend()
axes[0].grid(alpha=0.3)

for c, col in enumerate(cluster_colors):
    mask = labels == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=col, label=f'Cluster {c}', alpha=0.7, edgecolors='white')
centroids_pca = pca.transform(km_final.cluster_centers_)
axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                marker='X', s=200, color='black', zorder=5, label='Centroids')
axes[1].set_xlabel("PC 1")
axes[1].set_ylabel("PC 2")
axes[1].set_title("K-Means Clusters (PCA)")
axes[1].legend()

for i, (sp, col) in enumerate(zip(le.classes_, species_colors)):
    mask = true_labels == i
    axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=col, label=sp, alpha=0.7, edgecolors='white')
axes[2].set_xlabel("PC 1")
axes[2].set_ylabel("PC 2")
axes[2].set_title("True Species (PCA)")
axes[2].legend()

plt.tight_layout()
plt.savefig("outputs/L2_T3_kmeans_clustering.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to outputs/")
