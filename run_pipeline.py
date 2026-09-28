"""
Main entry point - runs the full gut-brain-axis multiomics pipeline
end to end: load -> preprocess -> integrate -> EDA -> dim. reduction
-> feature selection -> modeling.
"""

from src.data_loader import load_raw_datasets, filter_common_samples
from src.preprocessing import run_preprocessing, integrate_omics
from src.eda import plot_clinical_eda, plot_omics_distribution
from src.dimensionality_reduction import run_all_dim_reduction
from src.feature_selection import select_features
from src.modeling import train_and_evaluate_models, train_final_model_and_importances, run_rfecv


def main():
    # Step 1: Load and filter data
    clinical, gut_meta, saliva_meta, metabolomics = load_raw_datasets()
    clinical_f, gut_f, saliva_f, metab_f = filter_common_samples(
        clinical, gut_meta, saliva_meta, metabolomics
    )

    # Step 2: Preprocess + integrate
    gut_data, saliva_data, metab_data, clinical_data = run_preprocessing(
        clinical_f, gut_f, saliva_f, metab_f
    )
    full_data = integrate_omics(gut_data, saliva_data, metab_data, clinical_data)

    # Step 3: EDA
    plot_clinical_eda(full_data)
    plot_omics_distribution(gut_data, "Gut Microbiome")
    plot_omics_distribution(saliva_data, "Saliva Microbiome")
    plot_omics_distribution(metab_data, "Plasma Metabolomics")

    # Step 4: Dimensionality reduction
    run_all_dim_reduction(
        [(gut_data, "Gut"), (saliva_data, "Saliva"), (metab_data, "Metabolomics")],
        full_data["ADAS group"],
    )

    # Step 5: Feature selection
    final_features, corr_df, X, y = select_features(full_data)

    # Step 6: Modeling
    results_df, X_clean, y_encoded, le = train_and_evaluate_models(full_data, final_features)
    best_model, importances = train_final_model_and_importances(X_clean, y_encoded, le)

    # Step 7 (optional): RFECV for further refinement
    optimal_features = run_rfecv(X_clean, y_encoded)

    print("\nPipeline complete. Check results/figures/ and results/tables/ for outputs.")


if __name__ == "__main__":
    main()