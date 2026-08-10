import pytest
import numpy as np
import pandas as pd
from dpp_recommender.variant_generator.domain import VariantDataset
from dpp_recommender.descriptor_extraction import (
    extract_dpp,
    DPPProfile,
    InsufficientDataError,
    InvalidPredictorColumnsError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variant(df: pd.DataFrame, variant_id: str = "test-v") -> VariantDataset:
    """Create a minimal VariantDataset wrapper for testing."""
    return VariantDataset(
        dataframe=df,
        station_id="TEST",
        variant_id=variant_id,
        missingness_type="MCAR",
        missingness_level=0.0,
        random_seed=0,
    )


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """100-row DataFrame with no missing values."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {"tp": np.arange(100, dtype=float), "e": np.arange(100, dtype=float) * 2, "p": np.ones(100)},
        index=dates,
    )


PREDICTORS = ["tp", "e"]


# ===================================================================
# Missing Ratio
# ===================================================================

class TestMissingRatio:
    def test_zero_missing(self, clean_df):
        v = _make_variant(clean_df)
        profile = extract_dpp(v, PREDICTORS)
        assert profile.missing_ratio == 0.0

    def test_known_missing(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"tp": [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10],
             "e":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "p":  np.ones(10)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # 1 missing out of 20 predictor cells
        assert profile.missing_ratio == pytest.approx(1 / 20)

    def test_target_not_counted(self):
        """NaN in the target must not affect missing_ratio."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"tp": [1.0, 2, 3, 4, 5], "e": [1.0, 2, 3, 4, 5], "p": [np.nan] * 5},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.missing_ratio == 0.0


# ===================================================================
# Mean Gap Length
# ===================================================================

class TestMeanGapLength:
    def test_no_missing(self, clean_df):
        profile = extract_dpp(_make_variant(clean_df), PREDICTORS)
        assert profile.mean_gap_length == 0.0

    def test_single_isolated(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"tp": [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10],
             "e":  np.arange(10, dtype=float),
             "p":  np.ones(10)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # One gap of length 1 across all predictors → mean gap = 1, normalized = 1/10
        assert profile.mean_gap_length == pytest.approx(1 / 10)

    def test_one_consecutive_gap(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"tp": [1, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9, 10],
             "e":  np.arange(10, dtype=float),
             "p":  np.ones(10)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # One gap of length 3 → mean = 3, normalized = 3/10
        assert profile.mean_gap_length == pytest.approx(3 / 10)

    def test_multiple_gaps(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"tp": [1, np.nan, 3, np.nan, np.nan, 6, 7, 8, 9, 10],
             "e":  np.arange(10, dtype=float),
             "p":  np.ones(10)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # Two gaps: length 1 and length 2 → mean = 1.5, normalized = 1.5/10
        assert profile.mean_gap_length == pytest.approx(1.5 / 10)

    def test_gaps_separate_per_predictor(self):
        """Gaps in different predictors at the same row are separate gaps."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"tp": [np.nan, 2, 3, 4, 5],
             "e":  [1, 2, np.nan, 4, 5],
             "p":  np.ones(5)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # Two separate gaps (one per predictor), each length 1 → mean = 1, normalized = 1/5
        assert profile.mean_gap_length == pytest.approx(1 / 5)


# ===================================================================
# Lag-1 Autocorrelation
# ===================================================================

class TestLag1Autocorrelation:
    def test_known_series(self):
        """A perfectly linear series has autocorrelation ~1."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {"tp": np.arange(50, dtype=float),
             "e":  np.arange(50, dtype=float),
             "p":  np.ones(50)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.mean_lag1_autocorrelation == pytest.approx(1.0, abs=0.01)

    def test_missing_breaks_sequence(self):
        """A NaN between two values means those values are NOT a consecutive pair."""
        dates = pd.date_range("2023-01-01", periods=6, freq="D")
        # tp: 1, NaN, 3, 4, 5, 6  →  valid consecutive pairs: (3,4),(4,5),(5,6)
        # e:  identical
        df = pd.DataFrame(
            {"tp": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
             "e":  [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
             "p":  np.ones(6)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # Pairs (3,4),(4,5),(5,6) → linear → rho ≈ 1
        assert profile.mean_lag1_autocorrelation == pytest.approx(1.0, abs=0.01)

    def test_target_excluded(self, clean_df):
        """Target column should not affect lag-1 autocorrelation."""
        df1 = clean_df.copy()
        df2 = clean_df.copy()
        df2["p"] = np.random.default_rng(0).standard_normal(100)
        p1 = extract_dpp(_make_variant(df1), PREDICTORS)
        p2 = extract_dpp(_make_variant(df2), PREDICTORS)
        assert p1.mean_lag1_autocorrelation == pytest.approx(
            p2.mean_lag1_autocorrelation
        )

    def test_insufficient_data_raises(self):
        """If no predictor has enough valid consecutive pairs, raise."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"tp": [1.0, np.nan, np.nan],
             "e":  [np.nan, np.nan, 1.0],
             "p":  np.ones(3)},
            index=dates,
        )
        with pytest.raises(InsufficientDataError):
            extract_dpp(_make_variant(df), PREDICTORS)


# ===================================================================
# Trend Strength
# ===================================================================

class TestTrendStrength:
    def test_strong_linear(self):
        """A perfectly linear series → R² ≈ 1."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {"tp": np.arange(50, dtype=float),
             "e":  np.arange(50, dtype=float) * 3 + 7,
             "p":  np.ones(50)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.mean_trend_strength == pytest.approx(1.0, abs=0.01)

    def test_flat_series(self):
        """A near-constant (noisy) series → R² ≈ 0."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Small noise around a constant so lag-1 autocorrelation can still compute
        df = pd.DataFrame(
            {"tp": 5.0 + rng.normal(0, 0.001, 50),
             "e":  5.0 + rng.normal(0, 0.001, 50),
             "p":  np.ones(50)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.mean_trend_strength < 0.1

    def test_missing_values_ignored(self):
        """Trend still computed from observed values only."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        vals = np.arange(20, dtype=float)
        vals[5] = np.nan
        vals[10] = np.nan
        df = pd.DataFrame(
            {"tp": vals, "e": np.arange(20, dtype=float), "p": np.ones(20)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        # Still strongly linear
        assert profile.mean_trend_strength > 0.9

    def test_target_excluded(self, clean_df):
        df1 = clean_df.copy()
        df2 = clean_df.copy()
        df2["p"] = np.arange(100, dtype=float) * 999
        p1 = extract_dpp(_make_variant(df1), PREDICTORS)
        p2 = extract_dpp(_make_variant(df2), PREDICTORS)
        assert p1.mean_trend_strength == pytest.approx(p2.mean_trend_strength)

    def test_insufficient_raises(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"tp": [1.0, np.nan, np.nan],
             "e":  [np.nan, np.nan, np.nan],
             "p":  np.ones(3)},
            index=dates,
        )
        with pytest.raises(InsufficientDataError):
            extract_dpp(_make_variant(df), PREDICTORS)


# ===================================================================
# Pairwise Correlation
# ===================================================================

class TestPairwiseCorrelation:
    def test_highly_correlated(self):
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        vals = np.arange(50, dtype=float)
        df = pd.DataFrame(
            {"tp": vals, "e": vals * 2 + 5, "p": np.ones(50)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.mean_absolute_pairwise_correlation == pytest.approx(1.0, abs=0.01)

    def test_uncorrelated(self):
        """Two uncorrelated signals → correlation ≈ 0."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=1000, freq="D")
        df = pd.DataFrame(
            {"tp": rng.standard_normal(1000),
             "e":  rng.standard_normal(1000),
             "p":  np.ones(1000)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.mean_absolute_pairwise_correlation < 0.1

    def test_negative_correlation_absolute(self):
        """Negative correlation should produce the same |ρ| as positive."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        vals = np.arange(50, dtype=float)
        df = pd.DataFrame(
            {"tp": vals, "e": -vals, "p": np.ones(50)},
            index=dates,
        )
        profile = extract_dpp(_make_variant(df), PREDICTORS)
        assert profile.mean_absolute_pairwise_correlation == pytest.approx(1.0, abs=0.01)

    def test_target_excluded(self, clean_df):
        df1 = clean_df.copy()
        df2 = clean_df.copy()
        df2["p"] = np.arange(100, dtype=float)
        p1 = extract_dpp(_make_variant(df1), PREDICTORS)
        p2 = extract_dpp(_make_variant(df2), PREDICTORS)
        assert p1.mean_absolute_pairwise_correlation == pytest.approx(
            p2.mean_absolute_pairwise_correlation
        )

    def test_single_predictor_raises(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"tp": np.arange(10, dtype=float), "p": np.ones(10)},
            index=dates,
        )
        with pytest.raises(InsufficientDataError):
            extract_dpp(_make_variant(df), ["tp"])


# ===================================================================
# Overall DPP
# ===================================================================

class TestOverallDPP:
    def test_five_descriptors(self, clean_df):
        profile = extract_dpp(_make_variant(clean_df), PREDICTORS)
        vec = profile.to_vector()
        assert len(vec) == 5
        assert all(isinstance(v, float) for v in vec)

    def test_descriptor_order_stable(self, clean_df):
        profile = extract_dpp(_make_variant(clean_df), PREDICTORS)
        vec = profile.to_vector()
        assert vec[0] == profile.missing_ratio
        assert vec[1] == profile.mean_gap_length
        assert vec[2] == profile.mean_lag1_autocorrelation
        assert vec[3] == profile.mean_trend_strength
        assert vec[4] == profile.mean_absolute_pairwise_correlation

    def test_determinism(self, clean_df):
        v = _make_variant(clean_df)
        p1 = extract_dpp(v, PREDICTORS)
        p2 = extract_dpp(v, PREDICTORS)
        assert p1.to_vector() == p2.to_vector()

    def test_original_unchanged(self, clean_df):
        original = clean_df.copy()
        v = _make_variant(clean_df)
        extract_dpp(v, PREDICTORS)
        pd.testing.assert_frame_equal(clean_df, original)

    def test_invalid_predictor_raises(self, clean_df):
        with pytest.raises(InvalidPredictorColumnsError):
            extract_dpp(_make_variant(clean_df), ["nonexistent"])
