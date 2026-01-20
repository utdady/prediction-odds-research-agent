from pm_agent.normalization import normalize_yes_no


def test_normalize_yes_no_clip_and_arb_flag():
    r = normalize_yes_no(0.6, 0.45)
    assert r.illiquid_flag is True
    assert 0.001 <= r.p_norm <= 0.999


def test_normalize_yes_no_infer_missing():
    r = normalize_yes_no(0.4, None)
    assert abs(r.no_mid - 0.6) < 1e-9

