from pathlib import Path

from dpp_recommender.pipeline.run_descriptor_ablation_stability import (
    BOOTSTRAP_RESAMPLES,
    run_stability_analysis,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stability_analysis_records_paired_fold_metrics():
    summary = run_stability_analysis()
    assert summary["dataset"]["labelled_variant_count"] == 144
    assert summary["validation"]["number_of_evaluations"] == 25
    assert summary["validation"]["identical_splits_across_configurations"] is True
    assert summary["paired_bootstrap"]["resamples"] == BOOTSTRAP_RESAMPLES == 1000
    assert all(len(result["fold_metrics"]) == 25 for result in summary["configurations"].values())


def test_stability_analysis_paired_counts_and_full_values():
    summary = run_stability_analysis()
    full = summary["configurations"]["full_dpp"]
    assert abs(full["macro_f1"]["mean"] - 0.28178685095400124) < 1e-12
    assert abs(full["macro_f1"]["std"] - 0.06289948862353285) < 1e-12
    for name, result in summary["configurations"].items():
        if name == "full_dpp":
            continue
        for metric in ("macro_f1", "balanced_accuracy"):
            paired = result["paired_vs_full_dpp"][metric]
            assert paired["wins"] + paired["losses"] + paired["ties"] == 25
            assert len(result["fold_metrics"]) == 25


def test_stability_analysis_does_not_change_existing_outputs():
    paths = [
        REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_results.json",
        REPO_ROOT / "outputs" / "experiments" / "descriptor_ablation_results.csv",
        REPO_ROOT / "outputs" / "experiments" / "final_validation_report.json",
        REPO_ROOT / "outputs" / "experiments" / "robust_validation_report.json",
    ]
    before = {path: path.read_bytes() for path in paths}
    run_stability_analysis()
    assert all(path.read_bytes() == content for path, content in before.items())