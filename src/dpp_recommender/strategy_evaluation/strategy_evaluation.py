from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_STRATEGY_NAMES = [
    "drop_missing",
    "linear_interpolate",
    "ffill_bfill",
    "mean_impute",
    "iterative_impute",
]


def _split_train_test(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed dataframe into train and test partitions."""
    if len(dataframe) < 3:
        raise ValueError("At least 3 observations are required for strategy evaluation.")

    split_index = max(1, min(len(dataframe) - 1, int(round(len(dataframe) * 0.7))))
    return dataframe.iloc[:split_index].copy(), dataframe.iloc[split_index:].copy()


def _expanding_time_series_folds(dataframe: pd.DataFrame, n_folds: int = 3) -> list[dict]:
    """Create deterministic chronological expanding-window folds."""
    n_obs = len(dataframe)
    if n_obs < n_folds + 2:
        raise ValueError(
            f"At least {n_folds + 2} observations are required for {n_folds} chronological folds."
        )

    train_ends = [int(round(n_obs * frac)) for frac in np.linspace(0.5, 0.8, n_folds)]
    folds: List[dict] = []
    for fold_index, train_end in enumerate(train_ends, start=1):
        if train_end <= 1 or train_end >= n_obs:
            raise ValueError(
                "Generated fold boundaries are invalid for the current time series length."
            )
        folds.append(
            {
                "fold_id": f"fold_{fold_index}",
                "train_end": int(train_end),
                "test_start": int(train_end),
                "test_end": int(n_obs),
                "train_df": dataframe.iloc[:train_end].copy(),
                "test_df": dataframe.iloc[train_end:].copy(),
            }
        )
    return folds


def _aggregate_strategy_metrics(rows: List[dict]) -> dict:
    """Aggregate successful fold-level scores for a strategy."""
    rmse_values = [float(row["rmse"]) for row in rows if row["status"] == "success"]
    mae_values = [float(row["mae"]) for row in rows if row["status"] == "success"]
    r2_values = [float(row["r2"]) for row in rows if row["status"] == "success"]

    def _mean_std(values: List[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    mean_rmse, std_rmse = _mean_std(rmse_values)
    mean_mae, std_mae = _mean_std(mae_values)
    mean_r2, std_r2 = _mean_std(r2_values)
    return {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
    }


def _apply_strategy(
    dataframe: pd.DataFrame,
    predictors: Sequence[str],
    strategy_name: str,
    fit_values: Optional[dict] = None,
) -> pd.DataFrame:
    """Apply a preprocessing strategy to predictor columns without changing the target."""
    processed = dataframe.copy()

    if strategy_name == "drop_missing":
        return processed.dropna(subset=list(predictors)).copy()

    for predictor in predictors:
        if strategy_name == "linear_interpolate":
            processed[predictor] = processed[predictor].interpolate(method="linear", limit_direction="both")
        elif strategy_name == "ffill_bfill":
            processed[predictor] = processed[predictor].ffill().bfill()
        elif strategy_name == "mean_impute":
            if fit_values is None or predictor not in fit_values:
                raise ValueError("Mean imputation requires training-set means.")
            processed[predictor] = processed[predictor].fillna(fit_values[predictor])
        elif strategy_name == "iterative_impute":
            if fit_values is None:
                raise ValueError("Iterative imputation requires a fitted imputer.")
            processed[predictor] = processed[predictor].fillna(fit_values[predictor])
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

    return processed


def _prepare_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """Impute/train the strategy using only the training partition."""
    if strategy_name == "drop_missing":
        train_prepared = train_df.dropna(subset=list(predictors)).copy()
        test_prepared = test_df.dropna(subset=list(predictors)).copy()
        return train_prepared, test_prepared, None

    if strategy_name == "linear_interpolate":
        train_prepared = train_df.copy()
        test_prepared = test_df.copy()
        for predictor in predictors:
            train_prepared[predictor] = train_prepared[predictor].interpolate(method="linear", limit_direction="both")
            test_prepared[predictor] = test_prepared[predictor].interpolate(method="linear", limit_direction="both")
        return train_prepared, test_prepared, None

    if strategy_name == "ffill_bfill":
        train_prepared = train_df.copy()
        test_prepared = test_df.copy()
        for predictor in predictors:
            train_prepared[predictor] = train_prepared[predictor].ffill().bfill()
            test_prepared[predictor] = test_prepared[predictor].ffill().bfill()
        return train_prepared, test_prepared, None

    if strategy_name == "mean_impute":
        means = train_df[predictors].mean()
        train_prepared = train_df.copy()
        test_prepared = test_df.copy()
        for predictor in predictors:
            train_prepared[predictor] = train_prepared[predictor].fillna(means[predictor])
            test_prepared[predictor] = test_prepared[predictor].fillna(means[predictor])
        return train_prepared, test_prepared, {predictor: float(means[predictor]) for predictor in predictors}

    if strategy_name == "iterative_impute":
        imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False)
        imputer.fit(train_df[list(predictors)])
        train_prepared = train_df.copy()
        train_prepared[list(predictors)] = imputer.transform(train_df[list(predictors)])

        test_prepared = test_df.copy()
        test_prepared[list(predictors)] = imputer.transform(test_df[list(predictors)])
        return train_prepared, test_prepared, {"imputer": imputer}

    raise ValueError(f"Unsupported strategy: {strategy_name}")


def _evaluate_single_strategy_on_fold(
    variant_dataset,
    predictor_columns: Sequence[str],
    target_column: str,
    strategy_name: str,
    fold: dict,
) -> dict:
    """Evaluate a single strategy on one chronological fold."""
    train_df = fold["train_df"].copy()
    test_df = fold["test_df"].copy()

    train_prepared, test_prepared, _ = _prepare_split(train_df, test_df, predictor_columns, strategy_name)

    if train_prepared.empty or test_prepared.empty:
        raise ValueError(f"Strategy {strategy_name} produced an empty split for fold {fold['fold_id']}.")

    if len(train_prepared) < 2 or len(test_prepared) < 1:
        raise ValueError(f"Strategy {strategy_name} did not leave enough rows for training/testing in fold {fold['fold_id']}.")

    X_train = train_prepared[list(predictor_columns)]
    y_train = train_prepared[target_column]
    X_test = test_prepared[list(predictor_columns)]
    y_test = test_prepared[target_column]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "variant_id": variant_dataset.variant_id,
        "station_id": variant_dataset.station_id,
        "missingness_type": variant_dataset.missingness_type,
        "missingness_level": variant_dataset.missingness_level,
        "random_seed": variant_dataset.random_seed,
        "dataset_name": "FrenchPiezo",
        "strategy_name": strategy_name,
        "prediction_model": "LinearRegression",
        "fold_id": fold["fold_id"],
        "train_end": int(fold["train_end"]),
        "test_start": int(fold["test_start"]),
        "test_end": int(fold["test_end"]),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "status": "success",
        "failure_reason": None,
    }


def evaluate_strategy_benchmark(
    variants: Iterable,
    predictor_columns: Sequence[str],
    target_column: str,
    strategy_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Evaluate a compact benchmark of preprocessing strategies across chronological folds."""
    selected_strategies = list(strategy_names) if strategy_names is not None else DEFAULT_STRATEGY_NAMES
    all_rows: List[dict] = []

    for variant in variants:
        folds = _expanding_time_series_folds(variant.dataframe, n_folds=3)
        for strategy in selected_strategies:
            strategy_rows: List[dict] = []
            for fold in folds:
                try:
                    result = _evaluate_single_strategy_on_fold(
                        variant,
                        predictor_columns,
                        target_column,
                        strategy,
                        fold,
                    )
                    strategy_rows.append(result)
                except Exception as exc:  # explicit structured failure, never silent NaN
                    strategy_rows.append(
                        {
                            "variant_id": variant.variant_id,
                            "station_id": variant.station_id,
                            "missingness_type": variant.missingness_type,
                            "missingness_level": variant.missingness_level,
                            "random_seed": variant.random_seed,
                            "dataset_name": "FrenchPiezo",
                            "strategy_name": strategy,
                            "prediction_model": "LinearRegression",
                            "fold_id": fold["fold_id"],
                            "train_end": int(fold["train_end"]),
                            "test_start": int(fold["test_start"]),
                            "test_end": int(fold["test_end"]),
                            "rmse": float("nan"),
                            "mae": float("nan"),
                            "r2": float("nan"),
                            "status": "failed",
                            "failure_reason": str(exc),
                        }
                    )

            summary = _aggregate_strategy_metrics(strategy_rows)
            for row in strategy_rows:
                row.update(summary)
            all_rows.extend(strategy_rows)

    dataframe = pd.DataFrame(all_rows)
    return dataframe
