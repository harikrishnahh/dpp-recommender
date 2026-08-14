from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from dpp_recommender.dataset.ingest import load_french_piezo_station
from dpp_recommender.descriptor_extraction import extract_dpp
from dpp_recommender.variant_generator.generator import generate_variants
from dpp_recommender.strategy_evaluation import evaluate_strategy_benchmark
from dpp_recommender.meta_dataset.meta_dataset import build_meta_dataset, DPP_FEATURE_COLUMNS, _variant_label_frame
from dpp_recommender.pipeline.run_final_validation import _run_repeated_cv, get_leakage_audit, _get_metrics
from dpp_recommender.descriptor_extraction.exceptions import InsufficientDataError

REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 42

SELECTED_STATIONS = [
    "00365X0003/P1",
    "00061X0117/PZ1",
    "11221X0152/PZ1",
    "BSS002PTEJ/MONTFR",
    "BSS002PXUN/X",
    "BSS002PZXU/P",
    "BSS002QADR/X",
    "BSS003EFMA/X",
    "BSS003NYIW/X",
    "BSS003UGMK/X"
]

def bootstrap_bacc_macro_f1(y, dt_preds, maj_preds, label_order, n_bootstraps=1000):
    np.random.seed(RANDOM_SEED)
    bacc_dt, f1_dt = [], []
    bacc_maj, f1_maj = [], []
    
    y = np.array(y)
    dt_preds = np.array(dt_preds)
    maj_preds = np.array(maj_preds)
    
    n_samples = len(y)
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y[idx]
        dt_boot = dt_preds[idx]
        maj_boot = maj_preds[idx]
        
        # Guard against single class bootstrap folds
        if len(np.unique(y_boot)) < 2:
            continue
            
        bacc_dt.append(balanced_accuracy_score(y_boot, dt_boot))
        f1_dt.append(f1_score(y_boot, dt_boot, labels=label_order, average="macro", zero_division=0))
        
        bacc_maj.append(balanced_accuracy_score(y_boot, maj_boot))
        f1_maj.append(f1_score(y_boot, maj_boot, labels=label_order, average="macro", zero_division=0))
        
    def _ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "lower_95": float(np.percentile(arr, 2.5)),
            "upper_95": float(np.percentile(arr, 97.5))
        }
        
    return {
        "decision_tree": {
            "balanced_accuracy": _ci(bacc_dt),
            "macro_f1": _ci(f1_dt)
        },
        "majority": {
            "balanced_accuracy": _ci(bacc_maj),
            "macro_f1": _ci(f1_maj)
        }
    }


def _run_loso_strict(labels: pd.DataFrame):
    stations = sorted(labels["station_id"].dropna().unique().tolist())
    X = labels[DPP_FEATURE_COLUMNS].fillna(0.0)
    y = labels["best_strategy"]
    st = labels["station_id"]
    label_order = sorted(y.unique().tolist())
    
    results = {}
    for station in stations:
        train_mask = (st != station)
        test_mask = (st == station)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(y_train) == 0 or len(y_test) == 0:
            continue
            
        maj = y_train.value_counts().idxmax()
        preds_maj = [maj] * len(y_test)
        metrics_maj = _get_metrics(y_test, preds_maj, label_order)
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
        dt.fit(X_train, y_train)
        preds_dt = dt.predict(X_test)
        metrics_dt = _get_metrics(y_test, preds_dt, label_order)
        
        # Collect predictions for bootstrap later
        # Actually LOSO is done per station, no need to bootstrap LOSO, just the pooled CV predictions.
        
        missing_train = list(set(y_test.unique()) - set(y_train.unique()))
        
        results[str(station)] = {
            "test_size": int(len(y_test)),
            "classes_in_train": sorted(y_train.unique().tolist()),
            "classes_in_test": sorted(y_test.unique().tolist()),
            "absent_at_train_time": missing_train,
            "majority_macro_f1": metrics_maj["macro_f1"],
            "decision_tree_macro_f1": metrics_dt["macro_f1"],
        }
    return results

def _run_pooled_cv_and_bootstrap(labels: pd.DataFrame):
    X = labels[DPP_FEATURE_COLUMNS].fillna(0.0)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    
    all_y = []
    all_dt_preds = []
    all_maj_preds = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        maj = y_train.value_counts().idxmax()
        preds_maj = [maj] * len(y_test)
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
        dt.fit(X_train, y_train)
        preds_dt = dt.predict(X_test)
        
        all_y.extend(y_test.tolist())
        all_dt_preds.extend(preds_dt.tolist())
        all_maj_preds.extend(preds_maj)
        
    return bootstrap_bacc_macro_f1(all_y, all_dt_preds, all_maj_preds, label_order)

def get_dpp_correlations(meta_dataset: pd.DataFrame) -> dict:
    valid = meta_dataset.dropna(subset=["rmse"]).copy()
    variant_dpp = valid[["variant_id"] + DPP_FEATURE_COLUMNS].drop_duplicates(subset=["variant_id"])
    corr = variant_dpp[DPP_FEATURE_COLUMNS].corr().round(4).to_dict()
    return corr

def generate_multi_station_meta() -> pd.DataFrame:
    output_dir = REPO_ROOT / "outputs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = output_dir / "robust_meta_dataset_10_stations.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)
        
    dataset_path = REPO_ROOT / "data" / "raw" / "dataset_2015_2021.csv"
    
    all_variants = []
    
    print(f"Generating 10-station variant meta dataset...")
    for station in SELECTED_STATIONS:
        print(f" -> Station: {station}")
        station_df = load_french_piezo_station(dataset_path, station)
        st_variants = generate_variants(
            dataframe=station_df,
            station_id=station,
            missingness_levels=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            variants_per_level=8,
            random_seed=RANDOM_SEED,
        )
        all_variants.extend(st_variants)
        
    print(f"Total variants: {len(all_variants)}. Benchmarking...")
    benchmark = evaluate_strategy_benchmark(all_variants, ["tp", "e"], "p")
    
    print(f"Extracting profiles...")
    profile_map = {}
    valid_variants = []
    for variant in all_variants:
        try:
            profile_map[variant.variant_id] = extract_dpp(variant, ["tp", "e"])
            valid_variants.append(variant)
        except InsufficientDataError:
            print(f"Skipping variant {variant.variant_id} due to insufficient valid overlaps for correlation.")
            
    print(f"Valid variants remaining: {len(valid_variants)}")
    
    meta_dataset = build_meta_dataset(valid_variants, benchmark, profile_map, dataset_name="FrenchPiezo")
    meta_dataset.to_csv(meta_path, index=False)
    
    print(f"Saved to {meta_path}")
    return meta_dataset

def main():
    meta_dataset = generate_multi_station_meta()
    labels = _variant_label_frame(meta_dataset)

    # Note: station_id and missingness_level are already in labels from _variant_label_frame
    
    # Bootstrap CIs for Recommendation
    print("Running Pooled CV and Bootstrap...")
    bootstrap_results = _run_pooled_cv_and_bootstrap(labels)
    
    # LOSO
    print("Running LOSO...")
    loso = _run_loso_strict(labels)
    
    # Missing_ratio ablative compare via _run_repeated_cv
    print("Running Missing Ratio Baseline Compare...")
    from dpp_recommender.pipeline.run_final_validation import _run_repeated_cv
    tree_dpp = _run_repeated_cv(labels, lambda: DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED), DPP_FEATURE_COLUMNS, n_splits=5, n_repeats=5)
    tree_missing_ratio = _run_repeated_cv(labels, lambda: DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED), ["missing_ratio"], n_splits=5, n_repeats=5)
    
    # DPP Validation Deep Dive
    print("Generating Diagnostics...")
    corr = get_dpp_correlations(meta_dataset)
    from dpp_recommender.pipeline.run_final_validation import _dpp_diagnostics
    diagnostics = _dpp_diagnostics(meta_dataset)
    
    summary = {
        "dataset_summary": {
            "total_variants": len(labels),
            "stations_selected": SELECTED_STATIONS,
            "strategy_distribution": labels["best_strategy"].value_counts().to_dict()
        },
        "bootstrap_95_ci": bootstrap_results,
        "loso_strict_isolation": loso,
        "pooled_evaluation": {
            "full_dpp": {
                "macro_f1": tree_dpp["macro_f1"],
                "balanced_accuracy": tree_dpp["balanced_accuracy"]
            },
            "missing_ratio_only": {
                "macro_f1": tree_missing_ratio["macro_f1"],
                "balanced_accuracy": tree_missing_ratio["balanced_accuracy"]
            }
        },
        "dpp_diagnostics": {
            "correlation_matrix": corr,
            "descriptor_distributions": diagnostics
        },
        "leakage_audit": get_leakage_audit()
    }
    
    output_dir = REPO_ROOT / "outputs" / "experiments"
    report_path = output_dir / "robust_validation_report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    
    print("ROBUST VALIDATION COMPLETE")
    print(f"DecisionTree 95CI F1: {bootstrap_results['decision_tree']['macro_f1']}")

if __name__ == "__main__":
    main()
