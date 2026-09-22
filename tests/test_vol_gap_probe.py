import numpy as np

from finetune.vol_gap_probe import last_signal_days, summarize


def test_last_signal_days_keeps_the_latest_four():
    days = last_signal_days([10, 3, 10, 7, 8, 9], 4)
    assert days == [7, 8, 9, 10]


def test_exposure_gap_is_confirmed_only_when_ar_is_low_and_teacher_force_is_not():
    rows = []
    for _ in range(8):
        rows.append({
            "realized": 0.024,
            "ar_t065_n5": 0.012,
            "tf_t065_n5_weighted": 0.026,
            "tf_t065_n5_uniform": 0.030,
            "tf_t1_n16_weighted": 0.027,
        })
    summary = summarize(rows)
    assert summary["ar_t065_n5"]["calibration_ratio"] == 2.0
    assert summary["decision"]["confirmed_exposure_gap"] is True
    flat = [{**rows[0], "ar_t065_n5": 0.026}]
    assert summarize(flat)["decision"]["confirmed_exposure_gap"] is False
    assert np.isfinite(summary["tf_t1_n16_weighted"]["calibration_ratio"])
