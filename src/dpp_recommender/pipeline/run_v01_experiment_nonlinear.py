from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from dpp_recommender.meta_dataset import (
    evaluate_majority_baseline,
    evaluate_recommendation_baseline,
    evaluate_tree_baseline,
)
from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 42

def _dpp_variation_diagnostic(meta_dataset: pd.DataFrame) -> dict:
    """Calculates descriptive stats of DPP descriptors per strategy."""
    valid = meta_dataset.dropna(subset=["rmse"]).copy()
    if valid.empty:
        return {}

    per_variant = valid.groupby(["variant_id", "strategy_name"], as_index=False).agg(
        mean_rmse=("mean_rmse", "first")
    )
    best_idx = per_variant.groupby("variant_id")["mean_rmse"].idxmin()
    best_by_variant = per_variant.loc[best_idx, ["variant_id", "strategy_name"]]
    
    variant_dpp = valid[["variant_id"] + DPP_FEATURE_COLUMNS].drop_duplicates(subset=["variant_id"])
    merged = variant_dpp.merge(best_by_variant, on="variant_id", how="inner")
    
    diagnostic = {}
    for column in DPP_FEATURE_COLUMNS:
        diagnostic[column] = {}
        for strategy, group in merged.groupby("strategy_name"):
            diagnostic[column][str(strategy)] = {
                "mean": float(group[column].mean()),
                "median": float(group[column].median()),
                "std": float(group[column].std(ddof=0)) if len(group) > 1 else 0.0,
            }
            
    return diagnostic

def main() -> None:
    output_dir = REPO_ROOT / "outputs" / "experiments"
    meta_path = output_dir / "meta_dataset_expanded.csv"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta dataset at {meta_path}. Run the expanded experiment first.")
        
    meta_dataset = pd.read_csv(meta_path)
    
    # Run LogReg to get the split
    logreg_res = evaluate_recommendation_baseline(meta_dataset, random_state=RANDOM_SEED)
    train_variants = logreg_res["train_variants"]
    test_variants = logreg_res["test_variants"]
    
    # Run Majority baseline
    majority_res = evaluate_majority_baseline(
        meta_dataset,
        random_state=RANDOM_SEED,
        train_variants=train_variants,
        test_variants=test_variants,
    )
    
    # Run DecisionTree baseline
    tree_res = evaluate_tree_baseline(
        meta_dataset,
        random_state=RANDOM_SEED,
        train_variants=train_variants,
        test_variants=test_variants,
        max_depth=3,
        feature_columns=None,
    )
    
    # Ablations
    ablations = {}
    for feature in DPP_FEATURE_COLUMNS:
        res = evaluate_tree_baseline(
            meta_dataset,
            random_state=RANDOM_SEED,
            train_variants=train_variants,
            test_variants=test_variants,
            max_depth=3,
            feature_columns=[feature]
        )
        ablations[feature] = {
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "macro_f1": res["macro_f1"],
        }
    
    # Diagnostics
    diagnostics = _dpp_variation_diagnostic(meta_dataset)
    
    summary = {
        "number_of_training_variants": len(train_variants),
        "number_of_test_variants": len(test_variants),
        "strategy_class_distribution": tree_res["strategy_class_distribution"],
        "majority_baseline": {
            "accuracy": majority_res["accuracy"],
            "balanced_accuracy": majority_res["balanced_accuracy"],
            "macro_f1": majority_res["macro_f1"],
            "confusion_matrix": majority_res["confusion_matrix"],
            "per_class_metrics": majority_res["per_class_metrics"],
        },
        "logistic_regression": {
            "accuracy": logreg_res["recommendation_accuracy"],
            "balanced_accuracy": logreg_res["recommendation_balanced_accuracy"],
            "macro_f1": logreg_res["recommendation_macro_f1"],
            "confusion_matrix": logreg_res["confusion_matrix"],
            "per_class_metrics": logreg_res["per_class_metrics"],
        },
        "decision_tree_baseline": {
            "accuracy": tree_res["accuracy"],
            "balanced_accuracy": tree_res["balanced_accuracy"],
            "macro_f1": tree_res["macro_f1"],
            "confusion_matrix": tree_res["confusion_matrix"],
            "per_class_metrics": tree_res["per_class_metrics"],
        },
        "decision_tree_ablations": ablations,
        "dpp_feature_class_distribution_diagnostics": diagnostics,
        "label_order": tree_res["label_order"],
    }
    
    report_path = output_dir / "nonlinear_evaluation_report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    
    print(f"Nonlinear evaluation complete. Results saved to {report_path}.")
    print("\nSUMMARY OF MACRO F1 SCORES:")
    print(f"Majority:      {majority_res['macro_f1']:.4f}")
    print(f"LogReg:        {logreg_res['recommendation_macro_f1']:.4f}")
    print(f"DecisionTree:  {tree_res['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
