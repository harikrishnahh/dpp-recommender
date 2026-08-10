from dataclasses import dataclass
from typing import List


@dataclass
class DPPProfile:
    """
    Fixed-length descriptor vector characterizing preprocessing-relevant
    properties of a degraded multivariate time-series dataset.

    Version 0.1: exactly five descriptors. This set is frozen.
    """

    variant_id: str
    missing_ratio: float
    mean_gap_length: float
    mean_lag1_autocorrelation: float
    mean_trend_strength: float
    mean_absolute_pairwise_correlation: float

    def to_vector(self) -> List[float]:
        """Return the five descriptors as a fixed-order list."""
        return [
            self.missing_ratio,
            self.mean_gap_length,
            self.mean_lag1_autocorrelation,
            self.mean_trend_strength,
            self.mean_absolute_pairwise_correlation,
        ]
