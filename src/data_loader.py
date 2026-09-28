import pandas as pd
from src.config import DATA_DIR


def load_raw_datasets():
    """Load clinical, gut, saliva, and metabolomics data from data/raw/"""
    clinical = pd.read_csv(f"{DATA_DIR}/clinical.csv")
    gut_meta = pd.read_csv(f"{DATA_DIR}/gut_metagenomics.csv")
    saliva_meta = pd.read_csv(f"{DATA_DIR}/saliva_metagenomics.csv")
    metabolomics = pd.read_csv(f"{DATA_DIR}/metabolomics.csv")

    print("Clinical data shape:", clinical.shape)
    print("Gut metagenomics data shape:", gut_meta.shape)
    print("Saliva metagenomics data shape:", saliva_meta.shape)
    print("Metabolomics data shape:", metabolomics.shape)

    return clinical, gut_meta, saliva_meta, metabolomics


def filter_common_samples(clinical, gut_meta, saliva_meta, metabolomics):
    """Keep only patients present in all four datasets"""
    common_ids = (
        set(clinical["Patient ID"])
        & set(gut_meta["Patient ID"])
        & set(saliva_meta["Patient ID"])
        & set(metabolomics["Patient ID"])
    )
    print(f"Number of common samples across all omics layers: {len(common_ids)}")

    clinical_filtered = clinical[clinical["Patient ID"].isin(common_ids)].set_index("Patient ID")
    gut_filtered = gut_meta[gut_meta["Patient ID"].isin(common_ids)].set_index("Patient ID")
    saliva_filtered = saliva_meta[saliva_meta["Patient ID"].isin(common_ids)].set_index("Patient ID")
    metabolomics_filtered = metabolomics[metabolomics["Patient ID"].isin(common_ids)].set_index("Patient ID")

    print("\nFiltered data shapes:")
    print(f"Clinical data: {clinical_filtered.shape}")
    print(f"Gut metagenomics: {gut_filtered.shape}")
    print(f"Saliva metagenomics: {saliva_filtered.shape}")
    print(f"Metabolomics: {metabolomics_filtered.shape}")

    return clinical_filtered, gut_filtered, saliva_filtered, metabolomics_filtered