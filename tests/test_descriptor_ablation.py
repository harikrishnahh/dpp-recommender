import json
from pathlib import Path

from dpp_recommender.meta_dataset.meta_dataset import DPP_FEATURE_COLUMNS
from dpp_recommender.pipeline.run_descriptor_ablation import (
    EXPECTED_VARIANT_COUNT,
    FEATURE_CONFIGURATIONS,
    N_REPEATS,
    N_SPLITS,
    RANDOM_SEED,
    run_ablation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_ablation_configuration_contract():
    assert len(DPP_FEATURE_COLUMNS) == 5
    assert FEATURE_CONFIGURATIONS["full_dpp"] == DPP_FEATURE_COLUMNS
    assert all(len(columns) == 4 for name, columns in FEATURE_CONFIGURATIONS.items() if name != "full_dpp")


def test_descriptor_ablation_uses_v1_dataset_model_and_validation():
    summary = run_ablation()
    assert summary["dataset"]["labelled_variant_count"] == EXPECTED_VARIANT_COUNT == 144
    assert summary["model"] == {"name": "DecisionTreeClassifier", "max_depth": 3, "random_state": RANDOM_SEED}
    assert summary["validation"]["n_splits"] == N_SPLITS == 5
    assert summary["validation"]["n_repeats"] == N_REPEATS == 5
    assert set(summary["results"]) == {"majority", "missing_ratio_only", *FEATURE_CONFIGURATIONS}


def test_descriptor_ablation_reproduces_v1_baselines_and_preserves_outputs():
    summary = run_ablation()
    full = summary["results"]["full_dpp"]["macro_f1"]
    missing = summary["results"]["missing_ratio_only"]["macro_f1"]
    majority = summary["results"]["majority"]["macro_f1"]
    assert abs(full["mean"] - 0.2818) < 0.001
    assert abs(full["std"] - 0.0629) < 0.001
    assert abs(missing["mean"] - 0.1453) < 0.001
    assert abs(missing["std"] - 0.0114) < 0.001
    assert abs(majority["mean"] - 0.1485) < 0.001
    assert abs(majority["std"] - 0.0013) < 0.001

    original_paths = [
        REPO_ROOT / "outputs" / "experiments" / "final_validation_report.json",
        REPO_ROOT / "outputs" / "experiments" / "robust_validation_report.json",
        REPO_ROOT / "outputs" / "experiments" / "robust_meta_dataset_10_stations.csv",
        REPO_ROOT / "outputs" / "experiments" / "meta_dataset_expanded.csv",
    ]
    before = {path: path.read_bytes() for path in original_paths}
    assert all(path.exists() for path in original_paths)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_new_json_output_has_required_shape():
    output_path = REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_results.json"
    if not output_path.exists():
        return
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["dataset"]["labelled_variant_count"] == 144
    assert len(summary["dataset"]["descriptor_names"]) == 5
    assert all("macro_f1" in result for result in summary["results"].values())
    assert all("balanced_accuracy" in result for result in summary["results"].values())