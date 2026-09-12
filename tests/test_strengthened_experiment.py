import numpy as np
import pandas as pd

from dpp_recommender.descriptor_extraction import extract_dpp
from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS
from dpp_recommender.pipeline.run_strengthened_experiment import (
    CONVENTIONAL_FEATURE_COLUMNS,
    MIN_OBSERVATIONS,
    MIN_SPAN_DAYS,
    _fixed_and_oracle,
    _make_splits,
    _metric_dict,
    screen_stations,
)
from dpp_recommender.variant_generator import generate_variants


def test_station_screen_is_objective_and_records_exclusion_reason():
    dates = pd.date_range("2015-01-01", periods=MIN_OBSERVATIONS + 1, freq="D")
    good = pd.DataFrame({"bss": "GOOD", "time": dates, "tp": 1.0, "e": 2.0, "p": 3.0})
    short = good.iloc[:10].copy()
    short["bss"] = "SHORT"
    eligible, records = screen_stations(pd.concat([good, short], ignore_index=True))
    assert eligible["station_id"].tolist() == ["GOOD"]
    short_record = next(record for record in records if record["station_id"] == "SHORT")
    assert "minimum_observations" in short_record["exclusion_reasons"]
    assert "minimum_span_days" in short_record["exclusion_reasons"]


def test_strengthened_features_are_target_free_and_variant_ids_are_deterministic():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    frame = pd.DataFrame({"tp": np.arange(100.0), "e": np.arange(100.0) + 1, "p": np.arange(100.0) + 2}, index=dates)
    first = generate_variants(frame, "S", [0.2], 2, 42)
    second = generate_variants(frame, "S", [0.2], 2, 42)
    assert [variant.variant_id for variant in first] == [variant.variant_id for variant in second]
    profile = extract_dpp(first[0], ["tp", "e"])
    assert set(vars(profile)) == set(DPP_FEATURE_COLUMNS) | {"variant_id"}
    assert "p" not in DPP_FEATURE_COLUMNS
    assert set(CONVENTIONAL_FEATURE_COLUMNS) >= {"observation_count", "predictor_count", "missing_ratio"}


def test_shared_repeated_cv_split_manifest_is_aligned():
    labels = pd.DataFrame({
        "variant_id": [f"v{i}" for i in range(20)],
        "best_strategy": ["a"] * 10 + ["b"] * 10,
    })
    splits = _make_splits(labels)
    assert len(splits) == 25
    assert len({split["test_indices_hash"] for split in splits}) == 25
    assert all(split["train_count"] + split["test_count"] == 20 for split in splits)


def test_fixed_baseline_selects_strategy_from_training_variants_only():
    labels = pd.DataFrame({
        "variant_id": ["a", "b", "c", "d"],
        "station_id": ["S"] * 4,
        "best_strategy": ["drop_missing", "drop_missing", "mean_impute", "mean_impute"],
    })
    benchmark = pd.DataFrame({
        "variant_id": ["a", "b", "c", "d"] * 2,
        "strategy_name": ["drop_missing"] * 4 + ["mean_impute"] * 4,
        "mean_rmse": [1.0, 1.0, 10.0, 10.0, 2.0, 2.0, 0.5, 0.5],
    })
    splits = [{"fold_id": 0, "train": np.array([0, 1]), "test": np.array([2, 3])}]
    result = _fixed_and_oracle(labels, benchmark, splits)
    assert result["fixed_preprocessing"]["folds"][0]["selected_strategy"] == "drop_missing"
    assert result["fixed_preprocessing"]["folds"][0]["accuracy"] == 0.0