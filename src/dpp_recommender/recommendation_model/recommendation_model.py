from __future__ import annotations

from dpp_recommender.meta_dataset import fit_recommendation_model as _fit_recommendation_model


def fit_recommendation_model(meta_dataset):
    """Compatibility wrapper for the recommendation model entry point."""
    return _fit_recommendation_model(meta_dataset)
