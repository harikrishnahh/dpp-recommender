"""
Dataset ingestion module for the DPP-Recommender pipeline.
"""

from .ingest import (
    load_french_piezo_station,
    DatasetError,
    MissingColumnsError,
    StationNotFoundError,
    EmptyStationDataError,
)

__all__ = [
    "load_french_piezo_station",
    "DatasetError",
    "MissingColumnsError",
    "StationNotFoundError",
    "EmptyStationDataError",
]
