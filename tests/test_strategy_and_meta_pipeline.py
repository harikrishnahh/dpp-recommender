import pandas as pd
import numpy as np

from dpp_recommender.descriptor_extraction import extract_dpp
from dpp_recommender.meta_dataset import (
    build_meta_dataset,
    evaluate_majority_baseline,
    evaluate_recommendation_baseline,
    evaluate_tree_baseline,
    fit_recommendation_model,
)
from dpp_recommender.strategy_evaluation import (
    DEFAULT_STRATEGY_NAMES,
    _prepare_split,
    evaluate_strategy_benchmark,
)
from dpp_recommender.variant_generator.domain import VariantDataset


def _make_variant(variant_id: str = "variant-001") -> VariantDataset:
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "tp": np.linspace(1.0, 10.0, 40),
            "e": np.linspace(5.0, 15.0, 40),
            "p": np.linspace(20.0, 40.0, 40),
        },
        index=dates,
    )
    df.iloc[5, 0] = np.nan
    df.iloc[10, 1] = np.nan
    df.iloc[18, 0] = np.nan
    return VariantDataset(
        dataframe=df,
        station_id="00365X0003/P1",
        variant_id=variant_id,
        missingness_type="MCAR",
        missingness_level=0.05,
        random_seed=42,
    )


def test_strategy_benchmark_uses_chronological_expanding_folds():
    variant = _make_variant()
    results = evaluate_strategy_benchmark([variant], ["tp", "e"], "p")

    assert {"fold_id", "train_end", "test_start", "test_end"}.issubset(results.columns)
    assert results["fold_id"].nunique() == 3
    assert results["train_end"].nunique() == 3
    assert (results.groupby("fold_id")["train_end"].first().to_numpy() == np.array([20, 26, 32])).all()
    assert set(results["strategy_name"]) == set(DEFAULT_STRATEGY_NAMES)
    assert results["status"].isin(["success", "failed"]).all()
    assert results["mean_rmse"].notna().all()
    assert results["std_rmse"].notna().all()


def test_mean_imputation_is_fit_only_on_training_data():
    train_df = pd.DataFrame({"tp": [1.0, 2.0, np.nan, 4.0], "e": [5.0, np.nan, 7.0, 8.0], "p": [10.0, 11.0, 12.0, 13.0]})
    test_df = pd.DataFrame({"tp": [np.nan, 6.0], "e": [9.0, np.nan], "p": [14.0, 15.0]})

    train_prepared, test_prepared, _ = _prepare_split(train_df, test_df, ["tp", "e"], "mean_impute")

    assert train_prepared["tp"].notna().all()
    assert test_prepared["tp"].notna().all()
    assert np.isclose(train_prepared["tp"].mean(), 2.3333333333)
    assert np.isclose(train_prepared["e"].mean(), 6.6666666667)


def test_recommendation_split_is_variant_level_and_deterministic():
    variants = [_make_variant(f"variant-{i}") for i in range(1, 7)]
    benchmark = evaluate_strategy_benchmark(variants, ["tp", "e"], "p")
    profile_map = {v.variant_id: extract_dpp(v, ["tp", "e"]) for v in variants}
    meta = build_meta_dataset(variants, benchmark, profile_map)

    baseline_1 = evaluate_recommendation_baseline(meta)
    baseline_2 = evaluate_recommendation_baseline(meta)

    assert baseline_1["summary"]["number_of_training_variants"] > 0
    assert baseline_1["summary"]["number_of_test_variants"] > 0
    assert set(baseline_1["train_variants"]) & set(baseline_1["test_variants"]) == set()
    assert baseline_1["train_variants"] == baseline_2["train_variants"]
    assert baseline_1["test_variants"] == baseline_2["test_variants"]


def test_recommendation_model_roundtrip_and_summary_metrics():
    variants = [_make_variant(f"variant-{i}") for i in range(1, 7)]
    benchmark = evaluate_strategy_benchmark(variants, ["tp", "e"], "p")
    meta = build_meta_dataset(variants, benchmark, {v.variant_id: extract_dpp(v, ["tp", "e"]) for v in variants})

    baseline = fit_recommendation_model(meta)
    assert {
        "recommendation_accuracy",
        "recommendation_balanced_accuracy",
        "recommendation_macro_f1",
        "number_of_training_variants",
        "number_of_test_variants",
        "best_strategy",
    }.issubset(baseline)
    assert baseline["recommendation_accuracy"] >= 0.0
    assert baseline["number_of_training_variants"] > 0
    assert baseline["number_of_test_variants"] > 0


def test_majority_baseline_evaluation_uses_variant_level_split():
    variants = [_make_variant(f"variant-{i}") for i in range(1, 7)]
    benchmark = evaluate_strategy_benchmark(variants, ["tp", "e"], "p")
    meta = build_meta_dataset(variants, benchmark, {v.variant_id: extract_dpp(v, ["tp", "e"]) for v in variants})

    logistic = evaluate_recommendation_baseline(meta)
    majority = evaluate_majority_baseline(
        meta,
        train_variants=logistic["train_variants"],
        test_variants=logistic["test_variants"],
    )

    assert majority["number_of_training_variants"] == logistic["number_of_training_variants"]
    assert majority["number_of_test_variants"] == logistic["number_of_test_variants"]
    assert majority["label_order"] == logistic["label_order"]
    assert "macro_f1" in majority


def test_tree_baseline_evaluation_uses_variant_level_split():
    variants = [_make_variant(f"variant-{i}") for i in range(1, 7)]
    benchmark = evaluate_strategy_benchmark(variants, ["tp", "e"], "p")
    meta = build_meta_dataset(variants, benchmark, {v.variant_id: extract_dpp(v, ["tp", "e"]) for v in variants})

    logistic = evaluate_recommendation_baseline(meta)
    tree = evaluate_tree_baseline(
        meta,
        train_variants=logistic["train_variants"],
        test_variants=logistic["test_variants"],
        max_depth=3,
    )

    assert tree["number_of_training_variants"] == logistic["number_of_training_variants"]
    assert tree["number_of_test_variants"] == logistic["number_of_test_variants"]
    assert tree["label_order"] == logistic["label_order"]
    assert "macro_f1" in tree
    assert tree["model_name"] == "DecisionTreeClassifier"
