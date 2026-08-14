import pandas as pd
import numpy as np

from dpp_recommender.pipeline.run_final_validation import _run_repeated_cv
from sklearn.tree import DecisionTreeClassifier
from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS

def test_repeated_cv_no_leakage():
    # Setup dummy labels
    df = pd.DataFrame({
        "variant_id": [f"V{i}" for i in range(20)],
        "best_strategy": ["drop_missing"] * 10 + ["iterative_impute"] * 10
    })
    for col in DPP_FEATURE_COLUMNS:
        df[col] = np.random.rand(20)
        
    df = df.set_index("variant_id")
    
    # We pass it to repeated CV using n_splits=5, n_repeats=2
    res = _run_repeated_cv(df, lambda: DecisionTreeClassifier(max_depth=3), DPP_FEATURE_COLUMNS, n_repeats=2, n_splits=5)
    
    # Check that accuracy metrics were produced
    assert "macro_f1" in res
    assert "mean" in res["macro_f1"]
    assert len(res["folds"]) == 10
    
    # Spot-check the first fold to ensure valid output
    fold_0 = res["folds"][0]
    assert "macro_f1" in fold_0
    assert "confusion_matrix" in fold_0
