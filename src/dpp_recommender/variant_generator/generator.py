import uuid
import numpy as np
import pandas as pd
from typing import List

from .domain import VariantDataset
from .exceptions import InvalidMissingnessError, MissingPredictorError, EmptyDatasetError

def _inject_mcar(
    df: pd.DataFrame, 
    predictors: List[str], 
    missing_fraction: float, 
    seed: int
) -> pd.DataFrame:
    """
    Inject missing values completely at random across the combined pool 
    of specified predictors.
    """
    df_variant = df.copy()
    
    n_rows = len(df_variant)
    n_cols = len(predictors)
    total_elements = n_rows * n_cols
    
    n_missing = int(round(total_elements * missing_fraction))
    
    if n_missing == 0:
        return df_variant
        
    rng = np.random.default_rng(seed)
    
    # Randomly select flat indices from the combined predictor pool pool without replacement
    missing_indices = rng.choice(total_elements, size=n_missing, replace=False)
    
    # Map 1D flat index back to 2D row/col index in the predictor subarray
    row_indices, col_indices = np.unravel_index(missing_indices, (n_rows, n_cols))
    
    # Apply NaNs directly to the predictor subset 
    # Use to_numpy to avoid pandas loc ambiguity, update, and assign back
    pred_data = df_variant[predictors].to_numpy(dtype=float, copy=True)
    pred_data[row_indices, col_indices] = np.nan
    df_variant[predictors] = pred_data
    
    return df_variant

def generate_variants(
    dataframe: pd.DataFrame, 
    station_id: str,
    missingness_levels: List[float], 
    variants_per_level: int, 
    random_seed: int
) -> List[VariantDataset]:
    """
    Generates dataset variants incorporating MCAR missingness on predictor variables.
    
    Args:
        dataframe: The clean input dataframe indexed by datetime.
        station_id: The ID of the station in the input dataset.
        missingness_levels: List of proportions [0.0, 1.0] representing % of combined predictors missing value.
        variants_per_level: The amount of unique variants to generate per missingness level.
        random_seed: Absolute root random seed ensuring exact deterministic generation.
        
    Returns:
        List[VariantDataset]: A collection of variants matching constraints.
    """
    if dataframe.empty:
        raise EmptyDatasetError("Generation failed: Input dataframe is empty.")
        
    # The generator only processes explicit predictor variables
    predictors = ['tp', 'e']
    
    missing_cols = set(predictors) - set(dataframe.columns)
    if missing_cols:
        raise MissingPredictorError(
            f"Missing required predictor columns for generation: {missing_cols}"
        )
        
    for ml in missingness_levels:
        if not (0.0 <= ml <= 1.0):
            raise InvalidMissingnessError(
                f"Missingness level '{ml}' is invalid. Must be between 0.0 and 1.0."
            )
            
    # Derive exactly deterministic, uncorrelated child seeds using a base RNG
    base_rng = np.random.default_rng(random_seed)
    
    variants = []
    
    for ml in missingness_levels:
        for _ in range(variants_per_level):
            # Record explicit child seed representing exactly how this variant frame mutated
            variant_seed = int(base_rng.integers(0, 1_000_000_000))
            
            variant_df = _inject_mcar(dataframe, predictors, ml, variant_seed)
            
            # Using uuid for dataset keys 
            variant_id = str(uuid.uuid4())
            
            variants.append(VariantDataset(
                dataframe=variant_df,
                station_id=station_id,
                variant_id=variant_id,
                missingness_type="MCAR",
                missingness_level=ml,
                random_seed=variant_seed
            ))
            
    return variants
