"""
Descriptor extraction logic for the DPP-Recommender pipeline.

Computes the frozen five-descriptor DPP vector from a VariantDataset.
All calculations use predictor columns only, with no imputation.
"""

import numpy as np
import pandas as pd
from typing import List

from dpp_recommender.variant_generator.domain import VariantDataset
from .domain import DPPProfile
from .exceptions import InsufficientDataError, InvalidPredictorColumnsError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_dpp(
    variant: VariantDataset,
    predictor_columns: List[str],
) -> DPPProfile:
    """
    Extract the DPP descriptor vector from a VariantDataset.

    Args:
        variant: A VariantDataset produced by the variant generator stage.
        predictor_columns: Explicit list of predictor column names to use.
                           The target column must NOT be included.

    Returns:
        DPPProfile containing exactly five scalar descriptors.

    Raises:
        InvalidPredictorColumnsError: If any requested predictor is absent.
        InsufficientDataError: If statistics cannot be computed.
    """
    df = variant.dataframe

    # Validate predictor columns exist
    missing_cols = set(predictor_columns) - set(df.columns)
    if missing_cols:
        raise InvalidPredictorColumnsError(
            f"Predictor columns not found in DataFrame: {missing_cols}"
        )

    predictors = df[predictor_columns]

    return DPPProfile(
        variant_id=variant.variant_id,
        missing_ratio=_missing_ratio(predictors),
        mean_gap_length=_mean_gap_length(predictors),
        mean_lag1_autocorrelation=_mean_lag1_autocorrelation(predictors),
        mean_trend_strength=_mean_trend_strength(predictors),
        mean_absolute_pairwise_correlation=_mean_abs_pairwise_correlation(
            predictors
        ),
    )


# ---------------------------------------------------------------------------
# Descriptor 1: Missing Ratio
# ---------------------------------------------------------------------------

def _missing_ratio(predictors: pd.DataFrame) -> float:
    """
    M_missing = total NaN predictor cells / (T * D)

    Returns a value in [0, 1].
    """
    total_cells = predictors.shape[0] * predictors.shape[1]
    if total_cells == 0:
        return 0.0
    return float(predictors.isna().sum().sum() / total_cells)


# ---------------------------------------------------------------------------
# Descriptor 2: Normalized Mean Missing Gap Length
# ---------------------------------------------------------------------------

def _mean_gap_length(predictors: pd.DataFrame) -> float:
    """
    For each predictor, identify consecutive runs of NaN (gaps).
    Compute the mean gap length across ALL gaps from ALL predictors,
    then normalize by T (number of observations).

    Returns 0.0 if there are no missing values.
    """
    t = len(predictors)
    if t == 0:
        return 0.0

    all_gap_lengths: List[int] = []

    for col in predictors.columns:
        mask = predictors[col].isna().to_numpy()
        # Identify gap boundaries using diff of the boolean mask
        # A gap starts where mask transitions from False→True
        # A gap ends where mask transitions from True→False (or at end of series)
        in_gap = False
        current_length = 0
        for is_missing in mask:
            if is_missing:
                current_length += 1
                in_gap = True
            else:
                if in_gap:
                    all_gap_lengths.append(current_length)
                    current_length = 0
                    in_gap = False
        # Close any trailing gap
        if in_gap:
            all_gap_lengths.append(current_length)

    if len(all_gap_lengths) == 0:
        return 0.0

    return float(np.mean(all_gap_lengths) / t)


# ---------------------------------------------------------------------------
# Descriptor 3: Mean Lag-1 Autocorrelation
# ---------------------------------------------------------------------------

def _mean_lag1_autocorrelation(predictors: pd.DataFrame) -> float:
    """
    For each predictor, compute Pearson correlation between x_t and x_{t-1}
    using only valid consecutive pairs (both t-1 and t observed).
    Missing values break the sequence — no bridging across gaps.

    Returns the mean across predictors with valid statistics.
    Raises InsufficientDataError if no predictor has enough pairs.
    """
    correlations: List[float] = []

    for col in predictors.columns:
        series = predictors[col].to_numpy(dtype=float)
        # Build arrays of valid consecutive pairs
        x_prev: List[float] = []
        x_curr: List[float] = []
        for i in range(1, len(series)):
            if not np.isnan(series[i - 1]) and not np.isnan(series[i]):
                x_prev.append(series[i - 1])
                x_curr.append(series[i])

        # Need at least 2 pairs to compute a correlation
        if len(x_prev) < 2:
            continue

        prev_arr = np.array(x_prev)
        curr_arr = np.array(x_curr)

        # Check for zero variance (constant series)
        if np.std(prev_arr) == 0.0 or np.std(curr_arr) == 0.0:
            continue

        rho = np.corrcoef(prev_arr, curr_arr)[0, 1]
        correlations.append(float(rho))

    if len(correlations) == 0:
        raise InsufficientDataError(
            "No predictor has sufficient valid consecutive pairs "
            "to compute lag-1 autocorrelation."
        )

    return float(np.mean(correlations))


# ---------------------------------------------------------------------------
# Descriptor 4: Mean Trend Strength
# ---------------------------------------------------------------------------

def _mean_trend_strength(predictors: pd.DataFrame) -> float:
    """
    For each predictor, fit OLS (x = β₀ + β₁·t) on observed values only.
    Time variable is normalized to [0, 1] using the original chronological
    positions (not a compressed index after dropping NaN).
    R² is used as the trend-strength value.

    Returns the mean R² across valid predictors.
    Raises InsufficientDataError if no predictor is valid.
    """
    r_squared_values: List[float] = []
    t = len(predictors)

    if t <= 1:
        raise InsufficientDataError(
            "Need at least 2 observations to compute trend strength."
        )

    # Normalized time positions for ALL rows: [0, 1]
    time_positions = np.linspace(0.0, 1.0, t)

    for col in predictors.columns:
        series = predictors[col].to_numpy(dtype=float)
        valid_mask = ~np.isnan(series)

        # Need at least 2 observed values to fit a line
        if valid_mask.sum() < 2:
            continue

        t_valid = time_positions[valid_mask]
        x_valid = series[valid_mask]

        # Check for zero variance in time or values
        if np.std(t_valid) == 0.0 or np.std(x_valid) == 0.0:
            # Constant series or single unique time → R² undefined, treat as 0
            r_squared_values.append(0.0)
            continue

        # OLS via numpy: fit degree-1 polynomial
        coeffs = np.polyfit(t_valid, x_valid, 1)
        predicted = np.polyval(coeffs, t_valid)

        ss_res = np.sum((x_valid - predicted) ** 2)
        ss_tot = np.sum((x_valid - np.mean(x_valid)) ** 2)

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared_values.append(float(r2))

    if len(r_squared_values) == 0:
        raise InsufficientDataError(
            "No predictor has sufficient valid observations "
            "to compute trend strength."
        )

    return float(np.mean(r_squared_values))


# ---------------------------------------------------------------------------
# Descriptor 5: Mean Absolute Pairwise Correlation
# ---------------------------------------------------------------------------

def _mean_abs_pairwise_correlation(predictors: pd.DataFrame) -> float:
    """
    For every unique predictor pair (i, j), compute Pearson correlation
    using pairwise-complete observations, take the absolute value,
    and return the mean.

    Raises InsufficientDataError if fewer than 2 predictors exist.
    """
    cols = list(predictors.columns)
    n_cols = len(cols)

    if n_cols < 2:
        raise InsufficientDataError(
            "Need at least 2 predictor columns to compute pairwise correlation. "
            f"Got {n_cols}."
        )

    abs_correlations: List[float] = []

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            s_i = predictors[cols[i]].to_numpy(dtype=float)
            s_j = predictors[cols[j]].to_numpy(dtype=float)

            # Pairwise-complete: both must be observed
            valid = ~np.isnan(s_i) & ~np.isnan(s_j)

            if valid.sum() < 2:
                continue

            vals_i = s_i[valid]
            vals_j = s_j[valid]

            # Skip if either has zero variance
            if np.std(vals_i) == 0.0 or np.std(vals_j) == 0.0:
                continue

            rho = np.corrcoef(vals_i, vals_j)[0, 1]
            abs_correlations.append(abs(float(rho)))

    if len(abs_correlations) == 0:
        raise InsufficientDataError(
            "No valid predictor pair has sufficient pairwise-complete "
            "observations to compute correlation."
        )

    return float(np.mean(abs_correlations))
