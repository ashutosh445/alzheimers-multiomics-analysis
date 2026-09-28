import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from src.config import PROCESSED_DIR


def preprocess_data(df, id_col="Patient ID", impute_strategy="median", scale_method="standard"):
    """
    Preprocess omics data:
    1. Set Patient ID as index
    2. Handle missing values
    3. Remove low variance features
    4. Scale/normalize data
    """
    processed = df.copy()

    if id_col in processed.columns:
        processed = processed.set_index(id_col)

    # Remove features with >50% missing values
    threshold = len(processed) * 0.5
    processed = processed.dropna(thresh=threshold, axis=1)

    # Impute remaining missing values
    if impute_strategy == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=impute_strategy)

    processed_imputed = pd.DataFrame(
        imputer.fit_transform(processed),
        index=processed.index,
        columns=processed.columns,
    )

    # Remove low variance features (threshold = 0.1)
    variances = processed_imputed.var()
    processed_filtered = processed_imputed.loc[:, variances > 0.1]

    # Scale data
    scaler = MinMaxScaler() if scale_method == "minmax" else StandardScaler()
    processed_scaled = pd.DataFrame(
        scaler.fit_transform(processed_filtered),
        index=processed_filtered.index,
        columns=processed_filtered.columns,
    )

    return processed_scaled


def run_preprocessing(clinical_filtered, gut_filtered, saliva_filtered, metabolomics_filtered):
    """Preprocess all omics layers and save the results to data/processed/"""
    print("\nPreprocessing data...")

    gut_processed = preprocess_data(gut_filtered.reset_index())
    saliva_processed = preprocess_data(saliva_filtered.reset_index())
    metabolomics_processed = preprocess_data(metabolomics_filtered.reset_index())
    clinical_processed = clinical_filtered  # no scaling needed for clinical vars

    print("\nProcessed data shapes:")
    print(f"Gut metagenomics: {gut_processed.shape}")
    print(f"Saliva metagenomics: {saliva_processed.shape}")
    print(f"Metabolomics: {metabolomics_processed.shape}")
    print(f"Clinical: {clinical_processed.shape}")

    gut_processed.to_csv(f"{PROCESSED_DIR}/gut_processed_common_samples.csv")
    saliva_processed.to_csv(f"{PROCESSED_DIR}/saliva_processed_common_samples.csv")
    metabolomics_processed.to_csv(f"{PROCESSED_DIR}/metabolomics_processed_common_samples.csv")
    clinical_processed.to_csv(f"{PROCESSED_DIR}/clinical_processed_common_samples.csv")

    return gut_processed, saliva_processed, metabolomics_processed, clinical_processed


def integrate_omics(gut_data, saliva_data, metab_data, clinical_data):
    """Concatenate all omics layers with prefixed feature names + merge clinical data"""
    print("\n=== INTEGRATING MULTI-OMICS DATA ===")

    multi_omics = pd.concat(
        [
            gut_data.add_prefix("GUT_"),
            saliva_data.add_prefix("SALIVA_"),
            metab_data.add_prefix("METAB_"),
        ],
        axis=1,
    )

    full_data = pd.merge(multi_omics, clinical_data, left_index=True, right_index=True)
    full_data.to_csv(f"{PROCESSED_DIR}/FULL_MULTIOMICS_DATASET.csv")

    print("Integrated dataset dimensions:", full_data.shape)
    return full_data