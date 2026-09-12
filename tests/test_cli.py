from signalforge.cli import demo


def test_demo_is_deterministic_and_identifies_a_dominant_feature():
    first = demo(42)
    second = demo(42)
    assert first == second
    assert first["fingerprint"]["dominant_feature"] == "annual_spend"
    assert first["health"]["status"] in {"watch", "critical"}
