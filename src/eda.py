import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR


def plot_clinical_eda(full_data):
    """Clinical variable distributions across ADAS groups"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.boxplot(x="ADAS group", y="ADAS-COG", data=full_data)
    plt.title("Cognitive Scores by ADAS Group")

    plt.subplot(2, 2, 2)
    sns.countplot(x="SEX", hue="ADAS group", data=full_data)
    plt.title("Sex Distribution by ADAS Group")

    plt.subplot(2, 2, 3)
    sns.scatterplot(x="Age", y="ADAS-COG", hue="ADAS group", data=full_data)
    plt.title("Age vs Cognitive Score")

    plt.subplot(2, 2, 4)
    sns.heatmap(full_data[["ADAS-COG", "Age", "HB", "HCT", "RBC"]].corr(), annot=True)
    plt.title("Clinical Variable Correlations")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/CLINICAL_EDA.png")
    plt.show()


def plot_omics_distribution(data, title, top_n=20):
    """Boxplots of the top N most variable features in an omics layer"""
    variances = data.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[top_features])
    plt.title(f"Top {top_n} Variable Features - {title}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{title}_DISTRIBUTION.png")
    plt.show()