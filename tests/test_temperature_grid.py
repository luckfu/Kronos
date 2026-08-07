import pandas as pd
import pytest

from finetune.evaluate_temperature_grid import summarize_temperature


def test_summarize_temperature_adds_selection_metrics():
    frame = pd.DataFrame({
        'period': [0] * 5 + [1] * 5,
        'symbol': [f'symbol-{index}' for index in range(10)],
        'score': [0.5, 0.3, 0.1, -0.1, -0.4, 0.4, 0.2, 0.0, -0.2, -0.5],
        'actual_close_return': [0.10, 0.04, 0.01, -0.02, -0.08,
                                0.08, 0.03, 0.00, -0.01, -0.06],
    })

    summary, per_period = summarize_temperature(frame)

    assert summary['periods'] == 2
    assert summary['mean_top_bottom_actual_return_spread'] == pytest.approx(0.16)
    assert summary['predicted_down_rate'] == pytest.approx(0.4)
    assert len(per_period) == 2
