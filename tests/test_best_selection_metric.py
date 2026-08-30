import ast
from pathlib import Path


TRAINER = Path(
    "finetune/kaggle_v1_beta_minimal/Kronos/finetune/train_predictor.py"
)
CONFIG = Path("finetune/kaggle_v1_beta_minimal/Kronos/finetune/config.py")


def _function_source(path, name):
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_large_validation_can_select_best_without_quick_fallback():
    helper = _function_source(TRAINER, "best_selection_value")
    assert "validation_large_objective" in helper
    assert "large_metrics['objective_loss'] if large_metrics is not None else None" in helper


def test_large_validation_selection_is_a_guarded_config_value():
    source = CONFIG.read_text()
    assert 'KRONOS_BEST_SELECTION_METRIC' in source
    assert '"validation_large_objective"' in source
    trainer = TRAINER.read_text()
    assert "'best_selection_metric'," in trainer
    assert "selection_loss" in trainer


def test_full_only_validation_has_no_quick_metric_record():
    config = CONFIG.read_text()
    trainer = TRAINER.read_text()
    assert 'KRONOS_VALIDATION_FULL_ONLY' in config
    assert "'validation_full_only'," in trainer
    assert "if not validation_full_only:" in trainer
    assert "type='validation_large'" in trainer
    assert "quick_val_loader = None" in trainer
