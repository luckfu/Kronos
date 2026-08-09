import pandas as pd

from finetune.compare_kaggle_best_last import periods_with_lookback


def test_periods_with_lookback_trims_without_mutating_source():
    context = pd.DataFrame(
        {"close": range(120)},
        index=pd.date_range("2025-01-01", periods=120),
    )
    periods = [{"period": 0, "records": [{"symbol": "sh.600000", "context": context}]}]

    trimmed = periods_with_lookback(periods, 90)

    assert len(trimmed[0]["records"][0]["context"]) == 90
    assert trimmed[0]["records"][0]["context"].index[0] == context.index[30]
    assert len(periods[0]["records"][0]["context"]) == 120
