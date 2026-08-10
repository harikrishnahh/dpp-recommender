import pandas as pd
from pathlib import Path
from typing import Union

class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass

class MissingColumnsError(DatasetError):
    """Raised when required columns are missing from the dataset."""
    pass

class StationNotFoundError(DatasetError):
    """Raised when the specified station is not found in the dataset."""
    pass

class EmptyStationDataError(DatasetError):
    """Raised when the station data results in an empty dataframe after filtering."""
    pass

def load_french_piezo_station(filepath: Union[str, Path], station_id: str) -> pd.DataFrame:
    """
    Load the FrenchPiezo CSV, validate it, and extract data for a single station.

    Args:
        filepath: Path to the FrenchPiezo CSV file.
        station_id: The ID of the station to extract (must match the 'bss' column).

    Returns:
        pd.DataFrame: A clean DataFrame containing data for the specified station,
                      sorted and indexed by the parsed 'time' column.

    Raises:
        FileNotFoundError: If the file does not exist.
        MissingColumnsError: If the required columns are not present.
        StationNotFoundError: If the 'bss' column does not contain the specified station_id.
        EmptyStationDataError: If the resulting DataFrame for the station is empty.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Load dataset
    df = pd.read_csv(filepath)

    # 1. Validate required columns
    required_columns = {"bss", "time", "tp", "e", "p"}
    missing = required_columns - set(df.columns)
    if missing:
        raise MissingColumnsError(f"Missing required columns in dataset: {missing}")

    # 2. Validate station existence
    # We do this before filtering to provide a clear error message
    if station_id not in df["bss"].values:
        raise StationNotFoundError(f"Station ID '{station_id}' not found in the dataset.")

    # 3. Extract single station
    station_df = df[df["bss"] == station_id].copy()

    # 4. Check for empty dataframe (just in case all rows were somehow removed or empty)
    if station_df.empty:
        raise EmptyStationDataError(f"Data for station '{station_id}' is empty.")

    # 5. Convert time column to pandas datetime
    station_df["time"] = pd.to_datetime(station_df["time"])

    # 6. Sort by timestamp
    station_df = station_df.sort_values(by="time")

    # 7. Set 'time' as the index
    station_df = station_df.set_index("time")

    return station_df
