"""Fix BSS002PTEJ/MONTFR: exclude the single native NaN row (2015-02-28) in p,
regenerate its 48 variants, benchmark, and merge into the existing meta-dataset.

Does NOT regenerate or touch any other station's data.
Does NOT overwrite the original 2-station outputs (meta_dataset_expanded.csv, final_validation_report.json).
Saves new merged dataset as robust_meta_dataset_10_stations.csv (overwriting only the 10-station file).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from dpp_recommender.dataset.ingest import load_french_piezo_station
from dpp_recommender.descriptor_extraction import extract_dpp
from dpp_recommender.descriptor_extraction.exceptions import InsufficientDataError
from dpp_recommender.meta_dataset.meta_dataset import (
    DPP_FEATURE_COLUMNS,
    _variant_label_frame,
    build_meta_dataset,
)
from dpp_recommender.strategy_evaluation import evaluate_strategy_benchmark
from dpp_recommender.variant_generator.generator import generate_variants
from dpp_recommender.pipeline.run_final_validation import _get_metrics, _run_repeated_cv, get_leakage_audit

REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 42
DATASET_PATH = REPO_ROOT / "data" / "raw" / "dataset_2015_2021.csv"
STATION_ID = "BSS002PTEJ/MONTFR"
EXCLUDED_DATE = "2015-02-28"  # Native NaN in target p — data quality exclusion

OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments"
META_10_PATH = OUTPUT_DIR / "robust_meta_dataset_10_stations.csv"
REPORT_PATH = OUTPUT_DIR / "robust_validation_report.json"


def _bootstrap_paired_ci(y, dt_preds, maj_preds, label_order, n_bootstraps=1000):
    np.random.seed(RANDOM_SEED)
    y = np.array(y)
    dt_preds = np.array(dt_preds)
    maj_preds = np.array(maj_preds)
    n = len(y)

    f1_diffs, bacc_diffs = [], []
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        yb, dtb, majb = y[idx], dt_preds[idx], maj_preds[idx]
        if len(np.unique(yb)) < 2:
            continue
        f1_diffs.append(
            f1_score(yb, dtb, labels=label_order, average="macro", zero_division=0)
            - f1_score(yb, majb, labels=label_order, average="macro", zero_division=0)
        )
        bacc_diffs.append(
            balanced_accuracy_score(yb, dtb) - balanced_accuracy_score(yb, majb)
        )

    def _ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "lower_95": float(np.percentile(arr, 2.5)),
            "upper_95": float(np.percentile(arr, 97.5)),
        }

    return {"macro_f1_diff": _ci(f1_diffs), "balanced_accuracy_diff": _ci(bacc_diffs)}


def main():
    # ------------------------------------------------------------------ #
    # 1.  Load existing 10-station meta-dataset; drop old PTEJ rows        #
    # ------------------------------------------------------------------ #
    if not META_10_PATH.exists():
        raise FileNotFoundError(f"Expected {META_10_PATH}. Run run_robust_generalization.py first.")

    old_meta = pd.read_csv(META_10_PATH)
    other_stations_meta = old_meta[old_meta["station_id"] != STATION_ID].copy()
    print(f"Existing meta rows (other 2 stations): {len(other_stations_meta)}")

    # ------------------------------------------------------------------ #
    # 2.  Load PTEJ with the bad row excluded                              #
    # ------------------------------------------------------------------ #
    station_df = load_french_piezo_station(DATASET_PATH, STATION_ID)

    # Verify and apply exclusion
    nan_p_dates = station_df.index[station_df["p"].isna()]
    assert len(nan_p_dates) == 1, f"Expected exactly 1 NaN-p row, found {len(nan_p_dates)}: {nan_p_dates.tolist()}"
    assert str(nan_p_dates[0].date()) == EXCLUDED_DATE, f"Unexpected NaN date: {nan_p_dates[0]}"

    station_df_clean = station_df.drop(index=nan_p_dates)
    print(f"PTEJ rows before exclusion: {len(station_df)}")
    print(f"PTEJ rows after exclusion:  {len(station_df_clean)}")
    print(f"Excluded date: {EXCLUDED_DATE} (single native NaN in target p)")

    # ------------------------------------------------------------------ #
    # 3.  Generate 48 deterministic variants                               #
    # ------------------------------------------------------------------ #
    variants = generate_variants(
        dataframe=station_df_clean,
        station_id=STATION_ID,
        missingness_levels=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        variants_per_level=8,
        random_seed=RANDOM_SEED,
    )
    print(f"Generated {len(variants)} variants for {STATION_ID}")

    # ------------------------------------------------------------------ #
    # 4.  Strategy benchmark                                               #
    # ------------------------------------------------------------------ #
    print("Running strategy benchmark...")
    benchmark = evaluate_strategy_benchmark(variants, ["tp", "e"], "p")

    # ------------------------------------------------------------------ #
    # 5.  DPP extraction (skip variants with insufficient data)            #
    # ------------------------------------------------------------------ #
    print("Extracting DPP profiles...")
    profile_map = {}
    valid_variants = []
    for v in variants:
        try:
            profile_map[v.variant_id] = extract_dpp(v, ["tp", "e"])
            valid_variants.append(v)
        except InsufficientDataError:
            print(f"  Skipped {v.variant_id}: insufficient valid overlaps for correlation")

    print(f"  Valid PTEJ variants: {len(valid_variants)}")

    # ------------------------------------------------------------------ #
    # 6.  Build PTEJ-only meta-dataset segment                             #
    # ------------------------------------------------------------------ #
    ptej_meta = build_meta_dataset(valid_variants, benchmark, profile_map, dataset_name="FrenchPiezo")
    n_ptej_rmse_ok = ptej_meta["rmse"].notna().sum()
    n_ptej_rmse_fail = ptej_meta["rmse"].isna().sum()
    print(f"PTEJ benchmark: {n_ptej_rmse_ok} successful fold-evaluations, {n_ptej_rmse_fail} failed")

    # Strategy-level success breakdown
    for strat, grp in ptej_meta.groupby("strategy_name"):
        ok = grp["rmse"].notna().sum()
        fail = grp["rmse"].isna().sum()
        print(f"  {strat}: ok={ok}  fail={fail}")

    # ------------------------------------------------------------------ #
    # 7.  Merge with other stations and persist                            #
    # ------------------------------------------------------------------ #
    merged_meta = pd.concat([other_stations_meta, ptej_meta], ignore_index=True)
    merged_meta.to_csv(META_10_PATH, index=False)
    print(f"\nUpdated meta-dataset saved to {META_10_PATH}")
    print(f"Total rows: {len(merged_meta)}  |  stations: {sorted(merged_meta['station_id'].unique())}")

    # ------------------------------------------------------------------ #
    # 8.  Rebuild variant label frame and run LOSO + pooled CV             #
    # ------------------------------------------------------------------ #
    labels = _variant_label_frame(merged_meta)
    stations = sorted(labels["station_id"].dropna().unique().tolist())
    label_order = sorted(labels["best_strategy"].unique().tolist())
    X = labels[DPP_FEATURE_COLUMNS].fillna(0.0)
    y = labels["best_strategy"]

    print("\n--- LOSO RESULTS ---")
    loso_results = {}
    loso_f1s, loso_baccs = [], []
    for station in stations:
        train_mask = labels["station_id"] != station
        test_mask = labels["station_id"] == station
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if len(y_test) == 0:
            continue

        dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
        dt.fit(X_train, y_train)
        preds_dt = dt.predict(X_test)
        maj = y_train.value_counts().idxmax()
        preds_maj = [maj] * len(y_test)

        metrics_dt = _get_metrics(y_test, preds_dt, label_order)
        metrics_maj = _get_metrics(y_test, preds_maj, label_order)

        loso_f1s.append(metrics_dt["macro_f1"])
        loso_baccs.append(metrics_dt["balanced_accuracy"])
        loso_results[station] = {
            "test_variants": int(len(y_test)),
            "classes_in_train": sorted(y_train.unique().tolist()),
            "classes_in_test": sorted(y_test.unique().tolist()),
            "absent_at_train_time": sorted(set(y_test.unique()) - set(y_train.unique())),
            "decision_tree": {
                "accuracy": metrics_dt["accuracy"],
                "balanced_accuracy": metrics_dt["balanced_accuracy"],
                "macro_f1": metrics_dt["macro_f1"],
            },
            "majority": {
                "accuracy": metrics_maj["accuracy"],
                "balanced_accuracy": metrics_maj["balanced_accuracy"],
                "macro_f1": metrics_maj["macro_f1"],
            },
        }
        print(
            f"  {station}: n={len(y_test):3d}  "
            f"Acc={metrics_dt['accuracy']:.4f}  "
            f"BAcc={metrics_dt['balanced_accuracy']:.4f}  "
            f"F1={metrics_dt['macro_f1']:.4f}"
        )

    print(f"\nAggregate LOSO MacroF1:  {np.mean(loso_f1s):.4f}")
    print(f"Aggregate LOSO BAcc:     {np.mean(loso_baccs):.4f}")

    # ------------------------------------------------------------------ #
    # 9.  Pooled repeated CV + bootstrap CI                               #
    # ------------------------------------------------------------------ #
    print("\nRunning pooled CV...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    all_y, all_dt_preds, all_maj_preds = [], [], []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
        dt.fit(X_tr, y_tr)
        preds = dt.predict(X_te)
        maj = y_tr.value_counts().idxmax()
        all_y.extend(y_te.tolist())
        all_dt_preds.extend(preds.tolist())
        all_maj_preds.extend([maj] * len(y_te))

    ci = _bootstrap_paired_ci(all_y, all_dt_preds, all_maj_preds, label_order)

    # Pooled point estimates
    missing_ratio_cv = _run_repeated_cv(
        labels,
        lambda: DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
        ["missing_ratio"],
        n_splits=5,
        n_repeats=5,
    )
    full_dpp_cv = _run_repeated_cv(
        labels,
        lambda: DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
        DPP_FEATURE_COLUMNS,
        n_splits=5,
        n_repeats=5,
    )
    maj_cv = _run_repeated_cv(labels, None, DPP_FEATURE_COLUMNS, n_splits=5, n_repeats=5)

    # ------------------------------------------------------------------ #
    # 10. Persist updated report (does NOT overwrite original 2-station   #
    #     reports)                                                         #
    # ------------------------------------------------------------------ #
    report = {
        "metadata": {
            "description": "10-station robust validation with BSS002PTEJ/MONTFR fixed (2015-02-28 NaN-p row excluded)",
            "stations": stations,
            "total_label_variants": int(len(labels)),
            "ptej_fix": {
                "station": STATION_ID,
                "excluded_date": EXCLUDED_DATE,
                "reason": "Single native NaN in target variable p; data-quality exclusion only",
            },
            "strategy_distribution": labels["best_strategy"].value_counts().to_dict(),
        },
        "pooled_cv": {
            "majority": {"macro_f1": maj_cv["macro_f1"], "balanced_accuracy": maj_cv["balanced_accuracy"]},
            "full_dpp": {"macro_f1": full_dpp_cv["macro_f1"], "balanced_accuracy": full_dpp_cv["balanced_accuracy"]},
            "missing_ratio_only": {"macro_f1": missing_ratio_cv["macro_f1"], "balanced_accuracy": missing_ratio_cv["balanced_accuracy"]},
        },
        "bootstrap_paired_ci_dt_minus_majority": ci,
        "loso_strict_isolation": loso_results,
        "aggregate_loso": {
            "macro_f1_mean": float(np.mean(loso_f1s)),
            "balanced_accuracy_mean": float(np.mean(loso_baccs)),
        },
        "leakage_audit": get_leakage_audit(),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nUpdated report saved to {REPORT_PATH}")
    print("\n=== SUMMARY ===")
    print(f"Pooled Full-DPP MacroF1:        {full_dpp_cv['macro_f1']['mean']:.4f} ± {full_dpp_cv['macro_f1']['std']:.4f}")
    print(f"Pooled Majority MacroF1:        {maj_cv['macro_f1']['mean']:.4f} ± {maj_cv['macro_f1']['std']:.4f}")
    print(f"Pooled MissingRatio MacroF1:    {missing_ratio_cv['macro_f1']['mean']:.4f} ± {missing_ratio_cv['macro_f1']['std']:.4f}")
    print(f"Bootstrap CI (DT-Maj) F1:       [{ci['macro_f1_diff']['lower_95']:.4f}, {ci['macro_f1_diff']['upper_95']:.4f}]")
    print(f"Bootstrap CI (DT-Maj) BAcc:     [{ci['balanced_accuracy_diff']['lower_95']:.4f}, {ci['balanced_accuracy_diff']['upper_95']:.4f}]")


if __name__ == "__main__":
    main()
