from dpp_recommender.pipeline.run_descriptor_ablation_stability import run_stability_analysis


def test_paired_diagnostic_contains_requested_statistics_and_zero_exclusion():
    summary = run_stability_analysis()
    result = summary["configurations"]["without_mean_absolute_pairwise_correlation"]
    for metric in ("macro_f1", "balanced_accuracy"):
        paired = result["paired_vs_full_dpp"][metric]
        assert len(paired["paired_differences"]) == 25
        assert {"mean", "std", "median", "minimum", "maximum", "p25", "p75", "p95"}.issubset(paired)
        assert paired["wins"] + paired["losses"] + paired["ties"] == 25
        assert paired["ci_excludes_zero"] == (
            paired["bootstrap_ci_95"]["lower_95"] > 0
            or paired["bootstrap_ci_95"]["upper_95"] < 0
        )
