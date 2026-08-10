from dataclasses import dataclass
import pandas as pd

@dataclass
class VariantDataset:
    """
    Data model representing a specific dataset variant with missingness injected.
    Used for transferring data between pipeline stages.
    """
    dataframe: pd.DataFrame
    station_id: str
    variant_id: str
    missingness_type: str
    missingness_level: float
    random_seed: int
