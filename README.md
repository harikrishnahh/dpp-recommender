# DPP-Recommender

A framework that recommends preprocessing strategies for multivariate time-series datasets. 

This repository implements the research methodology for extracting a Dataset Preprocessing Profile (DPP), evaluating preprocessing strategies, building a meta-dataset, training a recommendation model, and predicting the best preprocessing strategy for unseen datasets.

## Research scope

This V0.1 implementation is intentionally constrained to the frozen five-descriptor DPP:

- missing_ratio
- mean_gap_length
- mean_lag1_autocorrelation
- mean_trend_strength
- mean_absolute_pairwise_correlation

The pipeline targets the FrenchPiezo dataset for station 00365X0003/P1, uses MCAR missingness injection on predictors tp and e, excludes the target p from DPP construction, and evaluates a small preprocessing benchmark with a fixed LinearRegression downstream model.

## Architecture

This project strictly adheres to clean, modular software engineering principles, emphasizing:
- **Simplicity over cleverness**
- **Modular design and Single Responsibility Principle**
- **Type hints and clear documentation**
- **Research reproducibility**

## Setup

```bash
# Optional: create a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate.ps1
# Mac/Linux: source venv/bin/activate

# Install dependencies in editable mode
pip install -e .[dev]
```

## Testing

Every module is independently testable. To run the test suite:
```bash
pytest
```

## Running the V0.1 experiment

To reproduce the real dataset + variant + DPP + strategy benchmark + recommendation workflow:

```bash
.\.venv\Scripts\python.exe src\dpp_recommender\pipeline\run_v01_experiment.py
```

This writes the following reproducible artifacts under the repository outputs directory:

- outputs/experiments/strategy_benchmark.csv
- outputs/experiments/meta_dataset.csv
- outputs/experiments/recommendation_summary.json

The script uses explicit deterministic settings for the MCAR variant generation and the recommendation split.
