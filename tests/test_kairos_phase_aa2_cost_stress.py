"""Unit tests for Phase AA2 cost stress helpers."""
from __future__ import annotations

from modernbert_finance.ablations.phase_aa2_cost_stress_before_paper import (
    COST_BPS_LEVELS,
    _bps_to_side,
    hard_conclusion,
)
from modernbert_finance.ablations.phase_aa2_kronos_longer_sealed_confirm import FOCUS_RULE


def test_cost_levels_and_conversion():
    assert COST_BPS_LEVELS == (20, 30, 40, 60)
    assert abs(_bps_to_side(30) - 0.003) < 1e-12
    assert abs(_bps_to_side(20) - 0.002) < 1e-12
    assert FOCUS_RULE == "topk50_tp10_nostop"


def test_hard_conclusion_paper_when_40_passes():
    def mk(pass_gate: bool, late_net: float, beats: bool = True):
        return {
            "pass_late_and_wf_gate": pass_gate,
            "z2_late_half": {
                "focus": {"mean_net_return": late_net, "sharpe_ann_net": 1.0,
                          "max_drawdown_net": -0.1, "hit_rate_net": 0.5},
                "random50_tp10_nostop": {"mean_net_return": -0.01},
                "buy_hold_ew": {"mean_net_return": -0.01},
                "focus_beats_random_and_ew": beats and late_net > 0,
                "a_priori_variants": {
                    "topk50_hold_d10": {"mean_net_return": late_net + 0.001},
                    "topk20_tp10_nostop": {"mean_net_return": late_net},
                },
            },
            "walkforward": {
                "fold_mean_nets": [late_net] * 4,
                "n_positive": 4 if late_net > 0 else 0,
                "n_folds": 4,
                "majority_positive": late_net > 0,
            },
        }

    results = {
        "levels": {
            "bps_20": mk(True, 0.02),
            "bps_30": mk(True, 0.014),
            "bps_40": mk(True, 0.008),
            "bps_60": mk(False, -0.001),
        }
    }
    c = hard_conclusion(results)
    assert c["recommend"] == "paper"
    assert 30 in c["still_pass_gate_bps"] and 40 in c["still_pass_gate_bps"]
    assert c["focus_rule"] == FOCUS_RULE
