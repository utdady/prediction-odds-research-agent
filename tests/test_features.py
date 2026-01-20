import pandas as pd


def test_feature_delta_computation_simple():
    p = pd.Series([0.4, 0.5, 0.55])
    dp = p.diff()
    assert abs(dp.iloc[1] - 0.1) < 1e-9

