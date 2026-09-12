import numpy as np
import pandas as pd

from dpp_recommender.pipeline.run_final_recommendation_analysis import (
    DEFAULT_STRATEGY_NAMES,
    NEAR_OPTIMAL_THRESHOLDS,
    _fixed_prediction,
    _multioutput_predict,
    _paired_ci,
    _variant_scores,
    recommendation_metrics,
)


def _small_inputs():
    features = pd.DataFrame({
        "variant_id": ["v1", "v2"],
        "station_id": ["s1", "s2"],
        "missingness_level": [0.05, 0.5],
        "random_seed": [1, 2],
        "missing_ratio": [0.05, 0.5],
        "mean_gap_length": [1.0, 2.0],
        "mean_lag1_autocorrelation": [0.1, 0.2],
        "mean_trend_strength": [0.3, 0.4],
        "mean_absolute_pairwise_correlation": [0.5, 0.6],
        "observation_count": [10, 10], "predictor_count": [2, 2], "missing_cell_count": [1, 2],
        "missing_ratio_sd_by_predictor": [0.0, 0.0], "predictor_std_mean": [1.0, 1.0],
        "predictor_skew_mean": [0.0, 0.0], "predictor_lag1_autocorrelation_mean": [0.1, 0.2],
        "predictor_pairwise_correlation_abs": [0.5, 0.6],
    })
    rows = []
    values = {"v1": [1.0, 1.001, 1.02, 1.1, 1.2], "v2": [2.0, 2.0, 2.0, 2.0, 2.0]}
    for variant_id, scores in values.items():
        for strategy, score in zip(DEFAULT_STRATEGY_NAMES, scores):
            rows.append({"variant_id": variant_id, "station_id": "s1" if variant_id == "v1" else "s2", "strategy_name": strategy, "mean_rmse": score})
    return features, pd.DataFrame(rows)


def test_variant_scores_calculates_oracle_margin_regrets_and_utility():
    features, benchmark = _small_inputs()
    labels, _ = _variant_scores(features, benchmark)
    v1 = labels.set_index("variant_id").loc["v1"]
    assert v1["oracle_strategy"] == "drop_missing"
    assert np.isclose(v1["margin"], 0.001)
    assert np.isclose(v1["absolute_regret_mean_impute"], 0.1)
    assert np.isclose(v1["relative_regret_mean_impute"], 0.1)
    assert np.isclose(v1["utility_drop_missing"], 1.0)
    assert labels.set_index("variant_id").loc["v2", "utility_mean_impute"] == 1.0


def test_recommendation_metrics_reports_near_optimal_thresholds():
    features, benchmark = _small_inputs()
    labels, _ = _variant_scores(features, benchmark)
    metrics = recommendation_metrics(labels.reset_index(drop=True), ["mean_impute", "drop_missing"])
    assert set(map(float, metrics["near_optimal_rates"])) == set(NEAR_OPTIMAL_THRESHOLDS)
    assert metrics["near_optimal_rates"]["0.05"] == 0.5
    assert metrics["absolute_regret"]["mean"] > 0.0


def test_fixed_baseline_uses_training_values_only():
    features, benchmark = _small_inputs()
    labels, _ = _variant_scores(features, benchmark)
    train = labels.iloc[[0]].copy()
    test = labels.iloc[[1]].copy()
    assert _fixed_prediction(train, test)[0] == "drop_missing"
    mutated = test.copy()
    mutated["rmse_drop_missing"] = 9999.0
    mutated["rmse_mean_impute"] = 0.0
    assert _fixed_prediction(train, mutated)[0] == "drop_missing"


def test_paired_ci_can_cluster_by_station():
    left = pd.Series([1.0, 2.0, 3.0, 4.0])
    right = pd.Series([0.0, 1.0, 2.0, 3.0])
    units = pd.Series(["s1", "s1", "s2", "s2"])
    result = _paired_ci(left, right, units)
    assert result["units"] == 2
    assert np.isclose(result["mean_difference"], 1.0)


def test_multioutput_prediction_uses_features_and_cached_training_targets():
    features, benchmark = _small_inputs()
    labels, _ = _variant_scores(features, benchmark)
    prediction = _multioutput_predict(
        labels.iloc[[0, 1]], labels.iloc[[0]], ["missing_ratio"], "utility",
        __import__("sklearn.tree", fromlist=["DecisionTreeRegressor"]).DecisionTreeRegressor(max_depth=3, random_state=42),
    )
    assert prediction[0] in DEFAULT_STRATEGY_NAMES
