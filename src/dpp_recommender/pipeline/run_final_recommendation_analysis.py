from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS
from dpp_recommender.pipeline.run_strengthened_experiment import (
    CONVENTIONAL_FEATURE_COLUMNS,
    MISSINGNESS_LEVELS,
    N_REPEATS,
    N_SPLITS,
    RANDOM_SEED,
)
from dpp_recommender.strategy_evaluation import DEFAULT_STRATEGY_NAMES


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments"
DATASET_PATH = OUTPUT_DIR / "strengthened_dataset.csv"
BENCHMARK_PATH = OUTPUT_DIR / "strengthened_strategy_benchmark.csv"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "strengthened_split_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "final_recommendation_analysis.json"
SUMMARY_CSV_PATH = OUTPUT_DIR / "final_recommendation_analysis.csv"
FOLD_RESULTS_PATH = OUTPUT_DIR / "final_recommendation_fold_results.csv"
STATION_RESULTS_PATH = OUTPUT_DIR / "final_station_results.csv"
NOVELTY_PATH = OUTPUT_DIR / "final_novelty_comparison.json"
BOOTSTRAP_RESAMPLES = 1000
NEAR_OPTIMAL_THRESHOLDS = [0.001, 0.005, 0.01, 0.05]
EXPECTED_STATIONS = 274
EXPECTED_VARIANTS = 13152


class CacheContractError(ValueError):
    pass


def _stable_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:24]


def load_cached_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only persisted strengthened artifacts; never invoke benchmark code."""
    for path in (DATASET_PATH, BENCHMARK_PATH, SPLIT_MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required strengthened artifact is missing: {path}")
    features = pd.read_csv(DATASET_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    required_features = {"variant_id", "station_id", "missingness_level", *DPP_FEATURE_COLUMNS, *CONVENTIONAL_FEATURE_COLUMNS}
    required_benchmark = {"variant_id", "station_id", "strategy_name", "mean_rmse", "status", "fold_id", "rmse"}
    if not required_features.issubset(features.columns):
        raise CacheContractError(f"Strengthened dataset lacks columns: {sorted(required_features - set(features.columns))}")
    if not required_benchmark.issubset(benchmark.columns):
        raise CacheContractError(f"Strengthened benchmark lacks columns: {sorted(required_benchmark - set(benchmark.columns))}")
    if len(features) != EXPECTED_VARIANTS or features["variant_id"].nunique() != EXPECTED_VARIANTS:
        raise CacheContractError(f"Expected {EXPECTED_VARIANTS} unique variant features, got {len(features)} rows")
    if features["station_id"].nunique() != EXPECTED_STATIONS:
        raise CacheContractError(f"Expected {EXPECTED_STATIONS} stations, got {features['station_id'].nunique()}")
    if set(benchmark["strategy_name"].unique()) != set(DEFAULT_STRATEGY_NAMES):
        raise CacheContractError("Benchmark strategy set does not match the five strengthened strategies")
    grouped = benchmark.groupby(["variant_id", "strategy_name"])
    if len(grouped) != EXPECTED_VARIANTS * len(DEFAULT_STRATEGY_NAMES):
        raise CacheContractError("Benchmark does not contain exactly five strategy rows per variant")
    if grouped["mean_rmse"].first().isna().any() or (grouped["status"].first() != "success").any():
        raise CacheContractError("Benchmark contains incomplete strategy outcomes")
    if len(manifest) != N_SPLITS * N_REPEATS:
        raise CacheContractError("Split manifest does not contain the expected 25 folds")
    return features, benchmark, manifest


def _variant_scores(features: pd.DataFrame, benchmark: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = benchmark.drop_duplicates(["variant_id", "strategy_name"])[
        ["variant_id", "station_id", "strategy_name", "mean_rmse"]
    ].copy()
    scores = aggregate.pivot(index="variant_id", columns="strategy_name", values="mean_rmse")
    scores = scores.reindex(columns=DEFAULT_STRATEGY_NAMES)
    if scores.isna().any().any():
        raise CacheContractError("Missing aggregate strategy score")
    scores.index = scores.index.astype(str)
    score_values = scores.to_numpy(dtype=float)
    oracle = np.min(score_values, axis=1)
    order = np.asarray([
        sorted(range(len(DEFAULT_STRATEGY_NAMES)), key=lambda index: (row[index], DEFAULT_STRATEGY_NAMES[index]))
        for row in score_values
    ])
    margins = score_values[np.arange(len(scores)), order[:, 1]] - oracle
    max_score = np.max(score_values, axis=1)
    span = max_score - oracle
    scaled_gap = np.divide(
        score_values - oracle[:, None],
        span[:, None],
        out=np.zeros_like(score_values),
        where=span[:, None] > 0,
    )
    utility = np.where(span[:, None] > 0, 1.0 - scaled_gap, 1.0)
    absolute_regret = score_values - oracle[:, None]
    relative_regret = np.divide(absolute_regret, oracle[:, None], out=np.full_like(absolute_regret, np.nan), where=oracle[:, None] > 1e-12)
    oracle_df = pd.DataFrame({
        "variant_id": scores.index,
        "oracle_strategy": [DEFAULT_STRATEGY_NAMES[i] for i in order[:, 0]],
        "oracle_rmse": oracle,
        "second_best_rmse": score_values[np.arange(len(scores)), order[:, 1]],
        "margin": margins,
        "relative_regret_guarded": (oracle <= 1e-12),
    }).set_index("variant_id")
    for index, strategy in enumerate(DEFAULT_STRATEGY_NAMES):
        oracle_df[f"rmse_{strategy}"] = score_values[:, index]
        oracle_df[f"absolute_regret_{strategy}"] = absolute_regret[:, index]
        oracle_df[f"relative_regret_{strategy}"] = relative_regret[:, index]
        oracle_df[f"utility_{strategy}"] = utility[:, index]
    labels = features.copy()
    labels["variant_id"] = labels["variant_id"].astype(str)
    derived_columns = [column for column in oracle_df.columns if column in labels.columns]
    labels = labels.drop(columns=derived_columns, errors="ignore")
    labels = labels.set_index("variant_id").join(oracle_df, how="inner").reset_index()
    labels = labels.sort_values("variant_id").reset_index(drop=True)
    labels["best_strategy"] = labels["oracle_strategy"]
    return labels, scores


def _summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": None, "median": None, "sd": None, "p25": None, "p75": None, "p95": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def recommendation_metrics(labels: pd.DataFrame, predicted: Iterable[str]) -> dict:
    predicted = np.asarray(list(predicted), dtype=object)
    actual = labels["oracle_strategy"].to_numpy(dtype=object)
    regrets = np.asarray([labels.iloc[i][f"absolute_regret_{strategy}"] for i, strategy in enumerate(predicted)], dtype=float)
    relative = np.asarray([labels.iloc[i][f"relative_regret_{strategy}"] for i, strategy in enumerate(predicted)], dtype=float)
    class_labels = sorted(labels["oracle_strategy"].unique().tolist())
    return {
        "absolute_regret": _summary(regrets),
        "relative_regret": _summary(relative),
        "zero_regret_rate": float(np.mean(regrets <= 1e-12)),
        "near_optimal_rates": {str(epsilon): float(np.mean(regrets <= epsilon)) for epsilon in NEAR_OPTIMAL_THRESHOLDS},
        "classification": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=class_labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        },
    }


def label_stability(labels: pd.DataFrame) -> dict:
    def describe(frame: pd.DataFrame) -> dict:
        margins = frame["margin"].to_numpy(dtype=float)
        return {"count": int(len(frame)), **_summary(margins), "threshold_fractions": {str(t): float(np.mean(margins <= t)) for t in [0.001, 0.005, 0.01, 0.05]}}
    return {
        "overall": describe(labels),
        "by_station": {str(k): describe(v) for k, v in labels.groupby("station_id")},
        "by_missingness_level": {str(k): describe(v) for k, v in labels.groupby("missingness_level")},
    }


def _make_pooled_splits(labels: pd.DataFrame) -> list[dict]:
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    y = labels["oracle_strategy"]
    splits = []
    for fold_id, (train, test) in enumerate(splitter.split(labels, y)):
        splits.append({"fold_id": fold_id, "repetition": fold_id // N_SPLITS, "fold_within_repetition": fold_id % N_SPLITS, "train": train, "test": test, "split_id": f"pooled_{fold_id:02d}"})
    return splits


def _check_manifest(labels: pd.DataFrame, splits: list[dict], manifest: pd.DataFrame) -> None:
    for split in splits:
        expected = manifest.loc[manifest["fold_id"] == split["fold_id"]].iloc[0]
        train_ids = labels.iloc[split["train"]]["variant_id"].astype(str).tolist()
        test_ids = labels.iloc[split["test"]]["variant_id"].astype(str).tolist()
        if _stable_hash(train_ids) != expected["train_indices_hash"] or _stable_hash(test_ids) != expected["test_indices_hash"]:
            raise CacheContractError(f"Reconstructed pooled split {split['fold_id']} differs from persisted manifest")


def _classifier_factories() -> dict[str, Callable[[], object]]:
    return {
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_SEED, n_jobs=1),
    }


def _fit_predict_classifier(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], factory: Callable[[], object]) -> np.ndarray:
    model = factory()
    model.fit(train[columns].fillna(0.0), train["oracle_strategy"])
    return model.predict(test[columns].fillna(0.0))


def _fixed_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    strategy = min(DEFAULT_STRATEGY_NAMES, key=lambda name: float(train[f"rmse_{name}"].mean()))
    return np.asarray([strategy] * len(test), dtype=object)


def _majority_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    strategy = train["oracle_strategy"].value_counts().sort_index().idxmax()
    return np.asarray([strategy] * len(test), dtype=object)


def _contextual_model(feature_columns: list[str], regressor) -> object:
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), feature_columns),
        ("strategy", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["candidate_strategy"]),
    ])
    return make_pipeline(transformer, regressor)


def _contextual_predict(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target_mode: str, regressor) -> np.ndarray:
    train_features = train[feature_columns].fillna(0.0).reset_index(drop=True)
    train_long = train_features.loc[train_features.index.repeat(len(DEFAULT_STRATEGY_NAMES))].reset_index(drop=True)
    train_long["candidate_strategy"] = np.tile(DEFAULT_STRATEGY_NAMES, len(train))
    target_columns = [f"utility_{s}" if target_mode == "utility" else f"rmse_{s}" for s in DEFAULT_STRATEGY_NAMES]
    train_long["target"] = train[target_columns].to_numpy(dtype=float).reshape(-1)
    model = _contextual_model(feature_columns, regressor)
    model.fit(train_long[feature_columns + ["candidate_strategy"]], train_long["target"])
    candidate_frame = test[feature_columns].fillna(0.0).reset_index(drop=True)
    candidate_frame = candidate_frame.loc[candidate_frame.index.repeat(len(DEFAULT_STRATEGY_NAMES))].reset_index(drop=True)
    candidate_frame["candidate_strategy"] = np.tile(DEFAULT_STRATEGY_NAMES, len(test))
    predictions = model.predict(candidate_frame[feature_columns + ["candidate_strategy"]]).reshape(len(test), len(DEFAULT_STRATEGY_NAMES))
    if target_mode == "utility":
        return np.asarray([DEFAULT_STRATEGY_NAMES[i] for i in np.argmax(predictions, axis=1)], dtype=object)
    return np.asarray([DEFAULT_STRATEGY_NAMES[i] for i in np.argmin(predictions, axis=1)], dtype=object)


def _multioutput_predict(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target_mode: str, estimator) -> np.ndarray:
    targets = [f"utility_{s}" if target_mode == "utility" else f"rmse_{s}" for s in DEFAULT_STRATEGY_NAMES]
    model = make_pipeline(StandardScaler(), MultiOutputRegressor(estimator))
    model.fit(train[feature_columns].fillna(0.0), train[targets])
    predictions = model.predict(test[feature_columns].fillna(0.0))
    choice = np.argmax(predictions, axis=1) if target_mode == "utility" else np.argmin(predictions, axis=1)
    return np.asarray([DEFAULT_STRATEGY_NAMES[i] for i in choice], dtype=object)


def _append_records(records: list[dict], labels: pd.DataFrame, predicted: Iterable[str], validation: str, split_id: str, representation: str, model_name: str, formulation: str) -> None:
    predicted = np.asarray(list(predicted), dtype=object)
    output = labels[["variant_id", "station_id", "missingness_level", "random_seed", "oracle_strategy", "oracle_rmse", "margin"]].copy()
    output["recommended_strategy"] = predicted
    output["recommended_rmse"] = [row[f"rmse_{strategy}"] for row, strategy in zip(labels.to_dict("records"), predicted)]
    output["absolute_regret"] = [row[f"absolute_regret_{strategy}"] for row, strategy in zip(labels.to_dict("records"), predicted)]
    output["relative_regret"] = [row[f"relative_regret_{strategy}"] for row, strategy in zip(labels.to_dict("records"), predicted)]
    for threshold in NEAR_OPTIMAL_THRESHOLDS:
        output[f"near_optimal_{threshold}"] = (output["absolute_regret"] <= threshold).astype(float)
    output.insert(0, "validation", validation)
    output.insert(1, "split_id", split_id)
    output.insert(2, "fold_id", split_id)
    output.insert(3, "representation", representation)
    output.insert(4, "model", model_name)
    output.insert(5, "formulation", formulation)
    records.extend(output.to_dict("records"))


def _summarize_records(records: pd.DataFrame) -> dict:
    output = {}
    for keys, group in records.groupby(["validation", "representation", "model", "formulation"], dropna=False):
        key = "|".join(map(str, keys))
        predicted = group["recommended_strategy"].to_numpy()
        absolute = group["absolute_regret"].to_numpy(dtype=float)
        relative = group["relative_regret"].to_numpy(dtype=float)
        output[key] = {
            "validation": keys[0], "representation": keys[1], "model": keys[2], "formulation": keys[3],
            "count": int(len(group)), "absolute_regret": _summary(absolute), "relative_regret": _summary(relative),
            "zero_regret_rate": float(np.mean(absolute <= 1e-12)),
            "near_optimal_rates": {str(t): float(np.mean(absolute <= t)) for t in NEAR_OPTIMAL_THRESHOLDS},
            "classification": {
                "accuracy": float(np.mean(predicted == group["oracle_strategy"].to_numpy())),
                "macro_f1": float(f1_score(group["oracle_strategy"], predicted, labels=DEFAULT_STRATEGY_NAMES, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(group["oracle_strategy"], predicted)),
            },
        }
    return output


def _paired_ci(left: pd.Series, right: pd.Series, units: pd.Series | None = None) -> dict:
    frame = pd.DataFrame({"left": left.to_numpy(float), "right": right.to_numpy(float), "unit": units.to_numpy() if units is not None else np.arange(len(left))})
    if units is not None:
        frame = frame.groupby("unit", as_index=False)[["left", "right"]].mean()
    differences = (frame["left"] - frame["right"]).to_numpy()
    if len(differences) == 0:
        return {
            "mean_difference": None,
            "median_difference": None,
            "lower_95": None,
            "upper_95": None,
            "units": 0,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        }
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
    means = differences[indices].mean(axis=1)
    return {"mean_difference": float(np.mean(differences)), "median_difference": float(np.median(differences)), "lower_95": float(np.percentile(means, 2.5)), "upper_95": float(np.percentile(means, 97.5)), "units": int(len(differences)), "bootstrap_resamples": BOOTSTRAP_RESAMPLES}


def _novelty_comparison() -> dict:
    return {
        "purpose": "Structured conceptual comparison; not a claim that prior preprocessing selection is absent.",
        "study_contribution": "Target-agnostic DPP descriptors select among causal preprocessing strategies for groundwater forecasting, using continuous strategy performance, regret, and unseen-station validation.",
        "comparisons": [
            {"work": "Pio et al. (2024), preprocessing algorithm selection review", "preprocessing_selection": "Reviews meta-learning and preprocessing selection", "meta_features": "Task/dataset meta-features vary by study", "target_access": "Varies", "missing_data_mechanism": "Not groundwater-specific", "time_series_handling": "Heterogeneous", "groundwater_specific": False, "interpretability": "Varies", "continuous_target": "Usually categorical or algorithm selection", "station_generalization": "Not the defining evaluation", "dpp_difference": "Groundwater-specific, target-agnostic missingness/temporal descriptors with regret evaluation."},
            {"work": "Baysal Erez et al. (2025), Extended MetaLIRS", "preprocessing_selection": "Meta-learning for preprocessing/algorithm selection", "meta_features": "Dataset meta-features", "target_access": "Benchmark-derived selection target", "missing_data_mechanism": "General tabular setting", "time_series_handling": "Not the central groundwater setting", "groundwater_specific": False, "interpretability": "Meta-learning dependent", "continuous_target": "Primarily selection performance", "station_generalization": "Not equivalent to leave-one-station-out groundwater evaluation", "dpp_difference": "Causal chronological forecasting folds, station-held-out inference, and continuous low-regret recommendation."},
            {"work": "Groundwater ML reviews and forecasting studies", "preprocessing_selection": "Usually prescribe or compare preprocessing within forecasting studies", "meta_features": "Hydrogeologic and time-series variables", "target_access": "Forecast target available to the forecasting model", "missing_data_mechanism": "Groundwater missingness context", "time_series_handling": "Commonly temporal forecasting", "groundwater_specific": True, "interpretability": "Varies", "continuous_target": "Forecast error, not usually preprocessing recommendation utility", "station_generalization": "Varies", "dpp_difference": "Recommendation layer isolates target-free incomplete-series characteristics from downstream target outcomes."},
            {"work": "Automated algorithm/meta-learning selection", "preprocessing_selection": "Algorithm or pipeline selection", "meta_features": "General meta-features", "target_access": "Task benchmark outcomes", "missing_data_mechanism": "Task-dependent", "time_series_handling": "Task-dependent", "groundwater_specific": False, "interpretability": "Varies", "continuous_target": "Often performance prediction or ranking", "station_generalization": "Depends on benchmark design", "dpp_difference": "Explicit five-strategy causal imputation choice, uncertainty through regret, and station-clustered inference."},
        ],
        "claims_supported": ["DPP reframes preprocessing recommendation around target-free incomplete time-series characteristics.", "The study tests practical regret rather than relying only on unstable exact winners.", "Unseen-station performance is evaluated as the primary generalization regime."],
    }


def _run_ablation(labels: pd.DataFrame, pooled_splits: list[dict], records: list[dict]) -> dict:
    results = {}
    for descriptor in DPP_FEATURE_COLUMNS:
        columns = [c for c in DPP_FEATURE_COLUMNS if c != descriptor]
        for split in pooled_splits:
            predicted = _fit_predict_classifier(labels.iloc[split["train"]], labels.iloc[split["test"]], columns, _classifier_factories()["decision_tree"])
            _append_records(records, labels.iloc[split["test"]].reset_index(drop=True), predicted, "pooled", split["split_id"], f"dpp_without_{descriptor}", "decision_tree", "exact_classification")
        results[f"without_{descriptor}"] = {"representation": f"dpp_without_{descriptor}", "feature_columns": columns}
    return results


def run_analysis() -> dict:
    started = time.perf_counter()
    features, benchmark, manifest = load_cached_inputs()
    labels, score_matrix = _variant_scores(features, benchmark)
    pooled_splits = _make_pooled_splits(labels)
    _check_manifest(labels, pooled_splits, manifest)
    records: list[dict] = []
    all_representation_columns = {"missing_ratio_only": ["missing_ratio"], "full_dpp": DPP_FEATURE_COLUMNS.copy(), "conventional_meta_features": CONVENTIONAL_FEATURE_COLUMNS.copy()}
    factories = _classifier_factories()
    for split in pooled_splits:
        train = labels.iloc[split["train"]].reset_index(drop=True)
        test = labels.iloc[split["test"]].reset_index(drop=True)
        _append_records(records, test, _majority_prediction(train, test), "pooled", split["split_id"], "majority", "majority", "exact_classification")
        _append_records(records, test, _fixed_prediction(train, test), "pooled", split["split_id"], "fixed_preprocessing", "fixed_preprocessing", "fixed_baseline")
        for representation, columns in all_representation_columns.items():
            for model_name, factory in factories.items():
                predicted = _fit_predict_classifier(train, test, columns, factory)
                _append_records(records, test, predicted, "pooled", split["split_id"], representation, model_name, "exact_classification")
            if representation in {"full_dpp", "missing_ratio_only", "conventional_meta_features"}:
                predicted = _contextual_predict(train, test, columns, "utility", DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED))
                _append_records(records, test, predicted, "pooled", split["split_id"], representation, "decision_tree_regressor", "contextual_normalized_utility")
                predicted = _multioutput_predict(train, test, columns, "utility", DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED))
                _append_records(records, test, predicted, "pooled", split["split_id"], representation, "multioutput_decision_tree", "multioutput_normalized_utility")
                predicted = _contextual_predict(train, test, columns, "rmse", Ridge(alpha=1.0))
                _append_records(records, test, predicted, "pooled", split["split_id"], representation, "ridge", "contextual_absolute_rmse")
    ablation_metadata = _run_ablation(labels, pooled_splits, records)
    for station_id in sorted(labels["station_id"].unique()):
        train = labels[labels["station_id"] != station_id].reset_index(drop=True)
        test = labels[labels["station_id"] == station_id].reset_index(drop=True)
        split_id = f"loso_{station_id}"
        _append_records(records, test, _majority_prediction(train, test), "station_grouped", split_id, "majority", "majority", "exact_classification")
        _append_records(records, test, _fixed_prediction(train, test), "station_grouped", split_id, "fixed_preprocessing", "fixed_preprocessing", "fixed_baseline")
        for representation, columns in all_representation_columns.items():
            predicted = _fit_predict_classifier(train, test, columns, factories["decision_tree"])
            _append_records(records, test, predicted, "station_grouped", split_id, representation, "decision_tree", "exact_classification")
            predicted = _contextual_predict(train, test, columns, "utility", DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED))
            _append_records(records, test, predicted, "station_grouped", split_id, representation, "decision_tree_regressor", "contextual_normalized_utility")
    frame = pd.DataFrame(records)
    summary = _summarize_records(frame)
    grouped = {}
    for key, group in frame.groupby(["validation", "representation", "model", "formulation"]):
        station_level = group.groupby("station_id")["absolute_regret"].mean()
        grouped["|".join(map(str, key))] = {
            "station_count": int(len(station_level)),
            "station_mean_regret": _summary(station_level),
            "station_near_optimal_rates": {str(t): float(np.mean(group.groupby("station_id")["absolute_regret"].mean() <= t)) for t in NEAR_OPTIMAL_THRESHOLDS},
        }
    primary = frame[(frame["validation"] == "station_grouped") & (frame["formulation"] == "contextual_normalized_utility")]
    comparison_results = {}
    for baseline in ["missing_ratio_only", "conventional_meta_features", "fixed_preprocessing"]:
        dpp = primary[primary["representation"] == "full_dpp"].set_index("variant_id")
        if baseline == "fixed_preprocessing":
            other = frame[
                (frame["validation"] == "station_grouped")
                & (frame["formulation"] == "fixed_baseline")
            ].set_index("variant_id")
        else:
            other = primary[primary["representation"] == baseline].set_index("variant_id")
        common = dpp.index.intersection(other.index)
        comparison_results[f"dpp_vs_{baseline}"] = _paired_ci(dpp.loc[common, "absolute_regret"], other.loc[common, "absolute_regret"], dpp.loc[common, "station_id"])
    missingness = {}
    for level in MISSINGNESS_LEVELS:
        subset = frame[(frame["validation"] == "station_grouped") & np.isclose(frame["missingness_level"], level)]
        missingness[str(level)] = _summarize_records(subset)
    output = {
        "study": {"name": "final_cached_low_regret_recommendation_study", "random_seed": RANDOM_SEED, "benchmark_recomputed": False, "runtime_seconds": float(time.perf_counter() - started)},
        "dataset": {"candidate_station_count": 2664, "eligible_station_count": int(labels["station_id"].nunique()), "variant_count": int(labels["variant_id"].nunique()), "variants_per_station": 48, "missingness_levels": MISSINGNESS_LEVELS, "strategies": DEFAULT_STRATEGY_NAMES, "forecast_model": "LinearRegression", "chronological_benchmark_folds": 3, "benchmark_failures": 0},
        "representations": all_representation_columns,
        "models": {"classification": list(factories), "continuous": ["decision_tree_regressor", "multioutput_decision_tree", "ridge"]},
        "target_formulation": {"primary": "within_variant_normalized_strategy_utility_for_training_and_absolute_rmse_regret_for_evaluation", "utility": "1 - (RMSE_s - min_s RMSE)/(max_s RMSE - min_s RMSE), utility=1 for all-tied variants", "secondary": "multioutput_normalized_utility", "sensitivity": "contextual_absolute_rmse", "absolute_regret": "RMSE_recommended - RMSE_oracle", "relative_regret": "absolute_regret / RMSE_oracle with undefined values guarded when oracle RMSE <= 1e-12", "justification": "Absolute regret is directly interpretable in groundwater RMSE units; relative regret supports cross-station scale comparison; normalized utility is bounded and stable for training."},
        "pooled_summary": {key: value for key, value in summary.items() if key.startswith("pooled|")},
        "station_grouped_summary": {key: value for key, value in summary.items() if key.startswith("station_grouped|")},
        "station_clustered_summary": grouped,
        "missingness_level_summary": missingness,
        "label_stability": label_stability(labels),
        "paired_primary_comparisons": comparison_results,
        "station_results_artifact": str(STATION_RESULTS_PATH.relative_to(REPO_ROOT)),
        "descriptor_ablation": ablation_metadata,
        "leakage_audit": {"dpp_uses_target": False, "station_id_used_as_feature": False, "held_out_outcomes_used_for_training": False, "held_out_outcomes_used_for_selection": False, "fixed_baseline_uses_training_only": True, "benchmark_source": "persisted strengthened cache/output"},
        "novelty_artifact": str(NOVELTY_PATH.relative_to(REPO_ROOT)),
    }
    return output, frame


def write_outputs(summary: dict, records: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    records.to_csv(FOLD_RESULTS_PATH, index=False)
    station_rows = []
    station_frame = records[records["validation"] == "station_grouped"]
    for keys, group in station_frame.groupby(["station_id", "representation", "model", "formulation"]):
        station_rows.append({
            "station_id": keys[0],
            "representation": keys[1],
            "model": keys[2],
            "formulation": keys[3],
            "variant_count": int(len(group)),
            "mean_absolute_regret": float(group["absolute_regret"].mean()),
            "median_absolute_regret": float(group["absolute_regret"].median()),
            "mean_relative_regret": float(group["relative_regret"].mean()),
            "near_optimal_0.001": float(group["near_optimal_0.001"].mean()),
            "near_optimal_0.005": float(group["near_optimal_0.005"].mean()),
            "near_optimal_0.01": float(group["near_optimal_0.01"].mean()),
            "near_optimal_0.05": float(group["near_optimal_0.05"].mean()),
            "exact_accuracy": float(np.mean(group["recommended_strategy"] == group["oracle_strategy"])),
        })
    pd.DataFrame(station_rows).to_csv(STATION_RESULTS_PATH, index=False)
    rows = []
    for key, value in summary["pooled_summary"].items():
        rows.append({"scope": "pooled", "key": key, "absolute_regret_mean": value["absolute_regret"]["mean"], "absolute_regret_median": value["absolute_regret"]["median"], **{f"near_optimal_{t}": value["near_optimal_rates"][t] for t in value["near_optimal_rates"]}, "classification_macro_f1": value["classification"]["macro_f1"]})
    for key, value in summary["station_grouped_summary"].items():
        rows.append({"scope": "station_grouped", "key": key, "absolute_regret_mean": value["absolute_regret"]["mean"], "absolute_regret_median": value["absolute_regret"]["median"], **{f"near_optimal_{t}": value["near_optimal_rates"][t] for t in value["near_optimal_rates"]}, "classification_macro_f1": value["classification"]["macro_f1"]})
    pd.DataFrame(rows).to_csv(SUMMARY_CSV_PATH, index=False)
    NOVELTY_PATH.write_text(json.dumps(_novelty_comparison(), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    summary, records = run_analysis()
    write_outputs(summary, records)
    print(json.dumps({"dataset": summary["dataset"], "runtime_seconds": summary["study"]["runtime_seconds"], "outputs": [str(path.relative_to(REPO_ROOT)) for path in (SUMMARY_PATH, SUMMARY_CSV_PATH, FOLD_RESULTS_PATH, STATION_RESULTS_PATH, NOVELTY_PATH)]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
