import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from src.config import RESULTS_DIR, FIGURES_DIR


def clean_feature_names(X):
    """Strip characters XGBoost chokes on"""
    X_clean = X.copy()
    X_clean.columns = [
        str(col).replace("[", "").replace("]", "").replace("<", "") for col in X_clean.columns
    ]
    return X_clean


def train_and_evaluate_models(full_data, final_features):
    """Train Random Forest, XGBoost, and Logistic Regression with 5-fold CV"""
    print("\n=== MACHINE LEARNING MODELING ===")

    X = full_data[final_features]
    y = full_data["ADAS group"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    X_clean = clean_feature_names(X)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_clean, y_encoded, cv=cv, scoring="accuracy")
        results.append({
            "Model": name,
            "Mean Accuracy": np.mean(scores),
            "Std Accuracy": np.std(scores),
        })
        print(f"{name}: Accuracy = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/MODEL_RESULTS.csv", index=False)

    return results_df, X_clean, y_encoded, le


def train_final_model_and_importances(X_clean, y_encoded, le):
    """Fit Random Forest on all selected features and rank importances"""
    best_model = RandomForestClassifier(random_state=42)
    best_model.fit(X_clean, y_encoded)

    importances = pd.DataFrame({
        "Feature": X_clean.columns,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importances["Category"] = importances["Feature"].apply(categorize_feature)
    importances.to_csv(f"{RESULTS_DIR}/FEATURE_IMPORTANCES.csv", index=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=importances.head(20))
    plt.title("Top 20 Important Features")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/TOP_FEATURES.png")
    plt.show()

    return best_model, importances


def categorize_feature(feature):
    if feature.startswith("GUT_"):
        return "Gut Microbiome"
    elif feature.startswith("SALIVA_"):
        return "Saliva Microbiome"
    elif feature.startswith("METAB_"):
        return "Metabolite"
    else:
        return "Clinical"


def run_rfecv(X_clean, y_encoded, min_features=20, step=10):
    """Recursive feature elimination with cross-validation"""
    selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=step,
        cv=StratifiedKFold(5),
        scoring="accuracy",
        min_features_to_select=min_features,
    )
    selector.fit(X_clean, y_encoded)

    optimal_features = X_clean.columns[selector.support_]
    print(f"Optimal number of features: {len(optimal_features)}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(selector.cv_results_["mean_test_score"]) + 1),
        selector.cv_results_["mean_test_score"],
    )
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean accuracy")
    plt.savefig(f"{FIGURES_DIR}/feature_selection_curve.png")
    plt.show()

    return optimal_features