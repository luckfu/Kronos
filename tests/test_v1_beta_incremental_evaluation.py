import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
PREPARE = ROOT / "finetune/prepare_v1_beta_incremental_evaluation.py"


def load_functions(names):
    tree = ast.parse(PREPARE.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    result = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PREPARE), "exec"), result)
    return result


def test_evaluation_package_name_uses_signal_cutoff_not_build_date():
    funcs = load_functions({"evaluation_package_name", "evaluation_package_title"})
    assert funcs["evaluation_package_name"]("2026-09-03") == (
        "kronos_beta_v2_time_oos_through_20260903"
    )
    assert funcs["evaluation_package_title"]("2026-09-03") == (
        "Kronos Beta v2 time-OOS evaluation through 2026-09-03"
    )
    source = PREPARE.read_text()
    assert "kronos_beta_v2_incremental_time_oos_20260902" not in source
    assert "evaluation_package_name(signal_end)" in source
