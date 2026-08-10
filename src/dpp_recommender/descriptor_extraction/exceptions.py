class DescriptorExtractionError(Exception):
    """Base exception for descriptor extraction errors."""
    pass


class InsufficientDataError(DescriptorExtractionError):
    """Raised when there are too few valid observations to compute a statistic."""
    pass


class InvalidPredictorColumnsError(DescriptorExtractionError):
    """Raised when requested predictor columns are not found in the DataFrame."""
    pass
