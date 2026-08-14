from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dpp_recommender.descriptor_extraction.domain import DPPProfile

DPP_FEATURE_COLUMNS = [
    "missing_ratio",
    "mean_gap_length",
    "mean_lag1_autocorrelation",
    "mean_trend_strength",
    "mean_absolute_pairwise_correlation",
]


def _coerce_profile_map(dpp_profiles: Iterable[DPPProfile] | Mapping[str, DPPProfile] | None) -> dict:
    if dpp_profiles is None:
        return {}
    if isinstance(dpp_profiles, Mapping):
        return {str(key): value for key, value in dpp_profiles.items()}
    return {str(profile.variant_id): profile for profile in dpp_profiles}


def build_meta_dataset(
    variants: Iterable,
    benchmark_results: pd.DataFrame,
    dpp_profiles: Iterable[DPPProfile] | Mapping[str, DPPProfile] | None = None,
    dataset_name: str = "FrenchPiezo",
) -> pd.DataFrame:
    """Join the benchmark results with DPP features and provenance metadata."""
    profile_map = _coerce_profile_map(dpp_profiles)
    rows: List[dict] = []

    benchmark = benchmark_results.copy()
    for variant in variants:
        variant_id = str(variant.variant_id)
        profile = profile_map.get(variant_id)
        variant_rows = benchmark[benchmark["variant_id"] == variant_id].copy()

        if profile is not None:
            profile_values = {column: float(getattr(profile, column)) for column in DPP_FEATURE_COLUMNS}
        else:
            profile_values = {column: float("nan") for column in DPP_FEATURE_COLUMNS}

        for _, row in variant_rows.iterrows():
            record = {
                "dataset_name": dataset_name,
                "station_id": getattr(variant, "station_id", row.get("station_id", None)),
                "variant_id": variant_id,
                "missingness_type": getattr(variant, "missingness_type", row.get("missingness_type", None)),
                "missingness_level": getattr(variant, "missingness_level", row.get("missingness_level", None)),
                "random_seed": getattr(variant, "random_seed", row.get("random_seed", None)),
                "strategy_name": row.get("strategy_name"),
                "prediction_model": row.get("prediction_model", "LinearRegression"),
                "rmse": row.get("rmse"),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "fold_id": row.get("fold_id"),
                "train_end": row.get("train_end"),
                "test_start": row.get("test_start"),
                "test_end": row.get("test_end"),
                "status": row.get("status", "success"),
                "failure_reason": row.get("failure_reason"),
                "mean_rmse": row.get("mean_rmse"),
                "std_rmse": row.get("std_rmse"),
                "mean_mae": row.get("mean_mae"),
                "std_mae": row.get("std_mae"),
                "mean_r2": row.get("mean_r2"),
                "std_r2": row.get("std_r2"),
            }
            record.update(profile_values)
            rows.append(record)

    if not rows:
        return pd.DataFrame(columns=[
            "dataset_name",
            "station_id",
            "variant_id",
            "missingness_type",
            "missingness_level",
            "random_seed",
            "strategy_name",
            "prediction_model",
            "fold_id",
            "train_end",
            "test_start",
            "test_end",
            "rmse",
            "mae",
            "r2",
            "status",
            "failure_reason",
            "mean_rmse",
            "std_rmse",
            "mean_mae",
            "std_mae",
            "mean_r2",
            "std_r2",
            *DPP_FEATURE_COLUMNS,
        ])

    return pd.DataFrame(rows)


def _best_strategy_by_variant(meta_dataset: pd.DataFrame) -> pd.Series:
    if meta_dataset.empty:
        raise ValueError("Meta dataset is empty; cannot fit a recommendation model.")
    valid = meta_dataset.copy()
    score_column = "mean_rmse" if "mean_rmse" in valid.columns else "rmse"
    valid = valid.dropna(subset=[score_column]).copy()
    if valid.empty:
        raise ValueError("Meta dataset contains no valid RMSE rows for recommendation fitting.")

    per_variant_strategy = valid.groupby(["variant_id", "strategy_name"], as_index=False).agg(
        strategy_score=(score_column, "first")
    )
    best_idx = per_variant_strategy.groupby("variant_id")["strategy_score"].idxmin()
    return per_variant_strategy.loc[best_idx, ["variant_id", "strategy_name"]].set_index("variant_id")["strategy_name"]


def _variant_label_frame(meta_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-variant label table with DPP features and best strategy labels."""
    if meta_dataset.empty:
        raise ValueError("Meta dataset is empty; cannot build recommendation labels.")

    valid = meta_dataset.dropna(subset=["rmse"]).copy()
    if valid.empty:
        raise ValueError("Meta dataset contains no valid rows; recommendation evaluation is impossible.")

    variant_features = valid.drop_duplicates(subset=["variant_id"]).copy()
    feature_columns = [column for column in DPP_FEATURE_COLUMNS if column in variant_features.columns]
    if not feature_columns:
        raise ValueError("Meta dataset does not include the required DPP feature columns.")

    best_by_variant = _best_strategy_by_variant(valid)
    labels = variant_features.set_index("variant_id").join(best_by_variant.rename("best_strategy"), how="left")
    labels["best_strategy"] = labels["best_strategy"].fillna(
        labels.get("strategy_name", pd.Series(index=labels.index, dtype=object))
    )
    return labels


def _split_variants(
    labels: pd.DataFrame,
    random_state: int,
    train_variants: Sequence[str] | None = None,
    test_variants: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Create or validate variant-level train/test splits."""
    if train_variants is not None and test_variants is not None:
        return list(train_variants), list(test_variants)

    y = labels["best_strategy"]
    variant_ids = labels.index.tolist()
    if y.nunique() > 1 and int(y.value_counts().min()) >= 2:
        train_ids, test_ids = train_test_split(
            variant_ids,
            test_size=0.3,
            random_state=random_state,
            stratify=y,
        )
    else:
        train_ids, test_ids = train_test_split(
            variant_ids,
            test_size=max(1, min(2, len(labels) // 3)),
            random_state=random_state,
        )

    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError("Variant-level recommendation split produced an empty train or test set.")

    return list(train_ids), list(test_ids)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, label_order: list[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )
    per_class = {
        str(label): {
            "precision": float(class_precision[idx]),
            "recall": float(class_recall[idx]),
            "f1": float(class_f1[idx]),
            "support": int(class_support[idx]),
        }
        for idx, label in enumerate(label_order)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(label_order) >= 2 else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class_metrics": per_class,
    }


def evaluate_recommendation_baseline(
    meta_dataset: pd.DataFrame,
    random_state: int = 42,
    train_variants: Sequence[str] | None = None,
    test_variants: Sequence[str] | None = None,
) -> dict:
    """Evaluate a variant-level recommendation baseline using a deterministic held-out split."""
    labels = _variant_label_frame(meta_dataset)

    feature_columns = [column for column in DPP_FEATURE_COLUMNS if column in labels.columns]
    X = labels[feature_columns].fillna(0.0)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    train_variants, test_variants = _split_variants(
        labels,
        random_state=random_state,
        train_variants=train_variants,
        test_variants=test_variants,
    )

    X_train = X.loc[train_variants]
    X_test = X.loc[test_variants]
    y_train = y.loc[train_variants]
    y_test = y.loc[test_variants]

    if y_train.nunique() < 2:
        model = None
        predictions = np.asarray([y_train.iloc[0]] * len(y_test), dtype=object)
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    actual_best_strategy = {str(variant_id): str(actual) for variant_id, actual in y_test.items()}
    predicted_strategy = {str(variant_id): str(predicted) for variant_id, predicted in zip(test_variants, predictions)}
    strategy_distribution = {str(key): int(value) for key, value in y.value_counts().to_dict().items()}
    metrics = _classification_metrics(y_test, np.asarray(predictions, dtype=object), label_order)

    summary = {
        "number_of_training_variants": int(len(train_variants)),
        "number_of_test_variants": int(len(test_variants)),
        "train_variants": list(map(str, train_variants)),
        "test_variants": list(map(str, test_variants)),
        "strategy_class_distribution": strategy_distribution,
        "best_strategy": y.value_counts().idxmax(),
    }

    result = {
        "model": model,
        "recommendation_accuracy": metrics["accuracy"],
        "recommendation_balanced_accuracy": metrics["balanced_accuracy"],
        "recommendation_macro_f1": metrics["macro_f1"],
        "number_of_training_variants": int(len(train_variants)),
        "number_of_test_variants": int(len(test_variants)),
        "train_variants": list(map(str, train_variants)),
        "test_variants": list(map(str, test_variants)),
        "strategy_class_distribution": strategy_distribution,
        "label_order": [str(label) for label in label_order],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class_metrics"],
        "predicted_strategy": predicted_strategy,
        "actual_best_strategy": actual_best_strategy,
        "best_strategy": summary["best_strategy"],
        "model_name": "LogisticRegression",
        "summary": summary,
    }

    return result


def evaluate_majority_baseline(
    meta_dataset: pd.DataFrame,
    random_state: int = 42,
    train_variants: Sequence[str] | None = None,
    test_variants: Sequence[str] | None = None,
) -> dict:
    """Evaluate a majority-class recommendation baseline on a variant-level split."""
    labels = _variant_label_frame(meta_dataset)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())

    train_variants, test_variants = _split_variants(
        labels,
        random_state=random_state,
        train_variants=train_variants,
        test_variants=test_variants,
    )
    y_train = y.loc[train_variants]
    y_test = y.loc[test_variants]

    majority_class = str(y_train.value_counts().idxmax())
    predictions = np.asarray([majority_class] * len(y_test), dtype=object)
    metrics = _classification_metrics(y_test, predictions, label_order)

    return {
        "baseline_name": "majority_class",
        "majority_class": majority_class,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class_metrics"],
        "number_of_training_variants": int(len(train_variants)),
        "number_of_test_variants": int(len(test_variants)),
        "train_variants": list(map(str, train_variants)),
        "test_variants": list(map(str, test_variants)),
        "label_order": [str(label) for label in label_order],
    }


def evaluate_tree_baseline(
    meta_dataset: pd.DataFrame,
    random_state: int = 42,
    train_variants: Sequence[str] | None = None,
    test_variants: Sequence[str] | None = None,
    max_depth: int = 3,
    feature_columns: list[str] | None = None,
) -> dict:
    """Evaluate a Decision Tree baseline to test for non-linear predictive capabilities."""
    labels = _variant_label_frame(meta_dataset)

    # Use all DPP features by default, or specific ablation features
    if feature_columns is None:
        feature_columns = [column for column in DPP_FEATURE_COLUMNS if column in labels.columns]
        
    X = labels[feature_columns].fillna(0.0)
    y = labels["best_strategy"]
    label_order = sorted(y.unique().tolist())
    train_variants, test_variants = _split_variants(
        labels,
        random_state=random_state,
        train_variants=train_variants,
        test_variants=test_variants,
    )

    X_train = X.loc[train_variants]
    X_test = X.loc[test_variants]
    y_train = y.loc[train_variants]
    y_test = y.loc[test_variants]

    if y_train.nunique() < 2:
        model = None
        predictions = np.asarray([y_train.iloc[0]] * len(y_test), dtype=object)
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    actual_best_strategy = {str(variant_id): str(actual) for variant_id, actual in y_test.items()}
    predicted_strategy = {str(variant_id): str(predicted) for variant_id, predicted in zip(test_variants, predictions)}
    strategy_distribution = {str(key): int(value) for key, value in y.value_counts().to_dict().items()}
    metrics = _classification_metrics(y_test, np.asarray(predictions, dtype=object), label_order)

    return {
        "model": model,
        "model_name": "DecisionTreeClassifier",
        "features_used": feature_columns,
        "max_depth": max_depth,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "number_of_training_variants": int(len(train_variants)),
        "number_of_test_variants": int(len(test_variants)),
        "train_variants": list(map(str, train_variants)),
        "test_variants": list(map(str, test_variants)),
        "strategy_class_distribution": strategy_distribution,
        "label_order": [str(label) for label in label_order],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class_metrics"],
        "predicted_strategy": predicted_strategy,
        "actual_best_strategy": actual_best_strategy,
    }



def fit_recommendation_model(meta_dataset: pd.DataFrame, random_state: int = 42) -> dict:
    """Standard V0.1 recommendation baseline with a variant-level held-out evaluation."""
    summary = evaluate_recommendation_baseline(meta_dataset, random_state=random_state)

    strategy_rankings = (
        meta_dataset.dropna(subset=["rmse"]).groupby("strategy_name")["rmse"].mean().sort_values().to_dict()
    )
    best_strategy = min(strategy_rankings, key=strategy_rankings.get) if strategy_rankings else None

    return {
        "model": summary["model"],
        "recommendation_accuracy": float(summary["recommendation_accuracy"]),
        "recommendation_balanced_accuracy": summary["recommendation_balanced_accuracy"],
        "recommendation_macro_f1": summary["recommendation_macro_f1"],
        "number_of_training_variants": int(summary["number_of_training_variants"]),
        "number_of_test_variants": int(summary["number_of_test_variants"]),
        "train_variants": summary["train_variants"],
        "test_variants": summary["test_variants"],
        "strategy_class_distribution": summary["strategy_class_distribution"],
        "label_order": summary["label_order"],
        "confusion_matrix": summary["confusion_matrix"],
        "per_class_metrics": summary["per_class_metrics"],
        "predicted_strategy": summary["predicted_strategy"],
        "actual_best_strategy": summary["actual_best_strategy"],
        "best_strategy": best_strategy,
        "strategy_accuracy": float(summary["recommendation_accuracy"]),
        "strategy_rankings": strategy_rankings,
    }
