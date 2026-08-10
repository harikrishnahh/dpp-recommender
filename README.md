# DPP-Recommender

A framework that recommends preprocessing strategies for multivariate time-series datasets. 

This repository implements the research methodology for extracting a Dataset Preprocessing Profile (DPP), evaluating preprocessing strategies, building a meta-dataset, training a recommendation model, and predicting the best preprocessing strategy for unseen datasets.

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
