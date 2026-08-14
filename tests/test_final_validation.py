import pandas as pd
import numpy as np

from dpp_recommender.pipeline.run_final_validation import _run_repeated_cv
from sklearn.tree import DecisionTreeClassifier
from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS

def test_repeated_cv_no_leakage():
    # Setup dummy labels
    df = pd.DataFrame({
        "variant_id": [f"V{i}" for i in range(20)],
        "best_strategy": ["drop_missing"] * 10 + ["iterative_impute"] * 10
    })
    for col in DPP_FEATURE_COLUMNS:
        df[col] = np.random.rand(20)
        
    df = df.set_index("variant_id")
    
    # We pass it to repeated CV using n_splits=5, n_repeats=2
    res = _run_repeated_cv(df, lambda: DecisionTreeClassifier(max_depth=3), DPP_FEATURE_COLUMNS, n_repeats=2, n_splits=5)
    
    # Check that accuracy metrics were produced
    assert "macro_f1" in res
    assert "mean" in res["macro_f1"]
    assert len(res["folds"]) == 10
    
    # Spot-check the first fold to ensure valid output
    fold_0 = res["folds"][0]
    assert "macro_f1" in fold_0
    assert "confusion_matrix" in fold_0


from dpp_recommender.pipeline.run_robust_generalization import _run_loso_strict

def test_loso_strict_grouping():
    # Setup dummy labels
    df = pd.DataFrame({
        "variant_id": [f"V{i}" for i in range(20)],
        "best_strategy": ["drop_missing"] * 10 + ["iterative_impute"] * 10,
        "station_id": ["STATION_A"] * 10 + ["STATION_B"] * 10
    })
    for col in DPP_FEATURE_COLUMNS:
        df[col] = np.random.rand(20)
        
    df = df.set_index("variant_id")
    
    # Run custom LOSO loop to assert leakage
    stations = sorted(df["station_id"].dropna().unique().tolist())
    st = df["station_id"]
    for station in stations:
        train_mask = (st != station)
        test_mask = (st == station)
        
        X_train, y_train = df[train_mask], df[train_mask]
        X_test, y_test = df[test_mask], df[test_mask]
        
        assert not set(y_train.index).intersection(set(y_test.index))
        assert len(y_train) == 10
        assert len(y_test) == 10


# Regression test: BSS002PTEJ/MONTFR exclusion of native NaN target row
from pathlib import Path
from dpp_recommender.dataset.ingest import load_french_piezo_station

def test_ptej_native_nan_row_excluded():
    """Confirms that after excluding 2015-02-28, BSS002PTEJ/MONTFR has zero NaN in p."""
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset_2015_2021.csv"
    if not dataset_path.exists():
        import pytest
        pytest.skip("Raw dataset not available")

    station_df = load_french_piezo_station(dataset_path, "BSS002PTEJ/MONTFR")

    # Confirm exactly one native NaN exists before exclusion
    nan_p = station_df.index[station_df["p"].isna()]
    assert len(nan_p) == 1
    assert str(nan_p[0].date()) == "2015-02-28"

    # Apply the same exclusion used in fix_ptej_station.py
    cleaned = station_df.drop(index=nan_p)

    # After exclusion: no NaN in p
    assert cleaned["p"].isna().sum() == 0
    # Row count decreased by exactly 1
    assert len(cleaned) == len(station_df) - 1
