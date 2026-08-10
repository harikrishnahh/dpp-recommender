import pytest
import pandas as pd
import numpy as np
from dpp_recommender.variant_generator import (
    generate_variants,
    VariantDataset,
    InvalidMissingnessError,
    MissingPredictorError,
    EmptyDatasetError
)

@pytest.fixture
def mock_station_data():
    """Mock dataset mimicking the output from Stage 1: dataset ingest."""
    dates = pd.date_range("2023-01-01", periods=100, freq='D')
    return pd.DataFrame({
        'tp': np.ones(100),
        'e': np.ones(100) * 2,
        'p': np.ones(100) * 3,
        'other': np.zeros(100)
    }, index=dates)

def test_generate_variants_success(mock_station_data):
    """Test standard generation logic on multiple variants and levels."""
    variants = generate_variants(mock_station_data, "STAT1", [0.1, 0.5], 2, 42)
    assert len(variants) == 4
    
    for v in variants:
        assert isinstance(v, VariantDataset)
        assert v.station_id == "STAT1"
        assert v.missingness_type == "MCAR"
        
        # Verify original wasn't modified
        assert not mock_station_data['tp'].isna().any()
        
        # Verify only tp and e have missing values, p is identically preserved
        assert not v.dataframe['p'].isna().any()
        assert not v.dataframe['other'].isna().any()
        
        # Verify combined pool missing proportion
        total_pool = len(mock_station_data) * 2
        total_missing = v.dataframe['tp'].isna().sum() + v.dataframe['e'].isna().sum()
        expected_missing = int(round(total_pool * v.missingness_level))
        assert total_missing == expected_missing
        
def test_generate_variants_determinism(mock_station_data):
    """Test reproducibility with fixed seeds and divergence with different seeds."""
    v1 = generate_variants(mock_station_data, "STAT1", [0.3], 1, 100)
    v2 = generate_variants(mock_station_data, "STAT1", [0.3], 1, 100)
    
    # 1. Output datasets are perfectly deterministic matching
    pd.testing.assert_frame_equal(v1[0].dataframe, v2[0].dataframe)
    # 2. Variants internally logged identically matching child seed 
    assert v1[0].random_seed == v2[0].random_seed
    
    v3 = generate_variants(mock_station_data, "STAT1", [0.3], 1, 999)
    # The different seed explicitly yields a different dataset permutation
    assert not v1[0].dataframe.equals(v3[0].dataframe)
    
def test_missing_predictors_validation(mock_station_data):
    """Missing predictor inputs fail fast."""
    df_bad = mock_station_data.drop(columns=['tp'])
    with pytest.raises(MissingPredictorError, match="Missing required predictor"):
        generate_variants(df_bad, "STAT1", [0.1], 1, 42)
        
def test_invalid_percentage_validation(mock_station_data):
    """Proportions bounds check raises correctly."""
    with pytest.raises(InvalidMissingnessError, match="Missingness level"):
        generate_variants(mock_station_data, "STAT1", [-0.5], 1, 42)
        
    with pytest.raises(InvalidMissingnessError, match="Missingness level"):
        generate_variants(mock_station_data, "STAT1", [1.5], 1, 42)
        
def test_empty_dataset_validation():
    """Empty datasets flag explicit error immediately."""
    df_empty = pd.DataFrame(columns=['tp', 'e', 'p'])
    with pytest.raises(EmptyDatasetError):
        generate_variants(df_empty, "STAT1", [0.1], 1, 42)
