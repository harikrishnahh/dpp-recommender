import pytest
import pandas as pd
from pathlib import Path
from dpp_recommender.dataset.ingest import (
    load_french_piezo_station,
    MissingColumnsError,
    StationNotFoundError,
    EmptyStationDataError
)

@pytest.fixture
def dummy_csv_path(tmp_path: Path) -> Path:
    """Fixture providing a valid dummy FrenchPiezo CSV."""
    df = pd.DataFrame({
        "bss": ["STATION_A", "STATION_B", "STATION_A"],
        "time": ["2023-01-02", "2023-01-01", "2023-01-01"],
        "tp": [1.1, 2.2, 3.3],
        "e": [10.1, 20.2, 30.3],
        "p": [100.1, 200.2, 300.3]
    })
    filepath = tmp_path / "french_piezo.csv"
    df.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def missing_columns_csv_path(tmp_path: Path) -> Path:
    """Fixture providing a CSV missing the 'tp' column."""
    df = pd.DataFrame({
        "bss": ["STATION_A"],
        "time": ["2023-01-01"],
        "e": [10.1],
        "p": [100.1]
    })
    filepath = tmp_path / "missing_cols.csv"
    df.to_csv(filepath, index=False)
    return filepath

def test_load_french_piezo_station_success(dummy_csv_path: Path):
    """Test successful loading, filtering, sorting, and indexing of a station."""
    df = load_french_piezo_station(dummy_csv_path, "STATION_A")
    
    # Assert correct shape (2 rows for STATION_A)
    assert len(df) == 2
    
    # Assert columns (time is now index, so it shouldn't be a column)
    assert "time" not in df.columns
    assert set(df.columns) == {"bss", "tp", "e", "p"}
    
    # Assert index is datetime and properly named
    assert df.index.name == "time"
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    
    # Assert sorting (2023-01-01 should come before 2023-01-02)
    assert df.index[0] == pd.Timestamp("2023-01-01")
    assert df.index[1] == pd.Timestamp("2023-01-02")
    
    # Assert all remaining rows belong to the specified station
    assert (df["bss"] == "STATION_A").all()

def test_missing_columns_error(missing_columns_csv_path: Path):
    """Test that MissingColumnsError is raised when columns are absent."""
    with pytest.raises(MissingColumnsError, match="Missing required columns"):
        load_french_piezo_station(missing_columns_csv_path, "STATION_A")

def test_station_not_found_error(dummy_csv_path: Path):
    """Test that StationNotFoundError is raised for a non-existent station."""
    with pytest.raises(StationNotFoundError, match="not found in the dataset"):
        load_french_piezo_station(dummy_csv_path, "STATION_Z")

def test_file_not_found_error():
    """Test that FileNotFoundError is raised for invalid paths."""
    with pytest.raises(FileNotFoundError):
        load_french_piezo_station("nonexistent/path/data.csv", "STATION_A")
