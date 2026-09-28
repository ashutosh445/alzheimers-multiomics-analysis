import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from src.config import FIGURES_DIR


def plot_reduced_dimensions(data, labels, title, method="pca"):
    """Visualize an omics layer in 2D using PCA, t-SNE, or UMAP"""
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="viridis", s=100)
    plt.title(f"{method.upper()} - {title}")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{method.upper()}_{title}.png")
    plt.show()


def run_all_dim_reduction(omics_layers, adas_labels):
    """
    omics_layers: list of (dataframe, name) tuples, e.g.
        [(gut_data, 'Gut'), (saliva_data, 'Saliva'), (metab_data, 'Metabolomics')]
    """
    print("\n=== DIMENSIONALITY REDUCTION ===")
    for data, name in omics_layers:
        for method in ["pca", "tsne", "umap"]:
            plot_reduced_dimensions(data, adas_labels, f"{name} Microbiome", method)