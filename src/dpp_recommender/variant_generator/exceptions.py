class VariantGenerationError(Exception):
    """Base exception for variant generator errors."""
    pass

class InvalidMissingnessError(VariantGenerationError):
    """Raised when the missingness level is outside the valid range [0, 1]."""
    pass

class MissingPredictorError(VariantGenerationError):
    """Raised when required predictor columns are not found in the input dataset."""
    pass

class EmptyDatasetError(VariantGenerationError):
    """Raised when the input dataframe is completely empty."""
    pass
