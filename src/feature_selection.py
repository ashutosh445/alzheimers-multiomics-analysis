import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, f_classif
from src.config import RESULTS_DIR


def select_features(full_data, k=50):
    """ANOVA + Spearman correlation feature selection, then take the union"""
    print("\n=== FEATURE SELECTION ===")

    X = full_data.drop(columns=["ADAS group", "ADAS-COG"])
    y = full_data["ADAS group"]

    # ANOVA
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    anova_features = X.columns[selector.get_support()]
    print(f"Selected {len(anova_features)} features by ANOVA")

    # Spearman correlation with ADAS-COG
    correlations = []
    for feature in X.columns:
        corr, _ = spearmanr(full_data[feature], full_data["ADAS-COG"])
        correlations.append(abs(corr))

    corr_df = pd.DataFrame({"Feature": X.columns, "Correlation": correlations})
    top_corr_features = corr_df.sort_values("Correlation", ascending=False).head(k)["Feature"].values

    # Union of both selection methods
    final_features = list(set(anova_features).union(set(top_corr_features)))
    print(f"Total unique features selected: {len(final_features)}")

    pd.DataFrame({"Selected_Features": final_features}).to_csv(
        f"{RESULTS_DIR}/SELECTED_FEATURES.csv", index=False
    )

    return final_features, corr_df, X, y