from plant_ood.smoke import run_synthetic_smoke_test


def test_synthetic_pipeline() -> None:
    output = run_synthetic_smoke_test()
    assert output["synthetic_only"] is True
    assert output["pipeline_shape"] == [12, 3]
    assert output["gate_shape"] == [12, 3]
    assert output["weights_normalized"] is True
