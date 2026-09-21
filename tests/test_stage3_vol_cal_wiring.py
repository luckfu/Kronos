from pathlib import Path

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / 'finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py'
TRAINER = ROOT / 'finetune/train_stage3_path_alignment.py'
MODEL = ROOT / 'finetune/stage3_training_model.py'


def test_smoke_wires_lambda_vol_and_vol_alignment_tests():
    source = SMOKE.read_text()
    for token in (
        "LAMBDA_VOL = float(os.environ.get('STAGE3_LAMBDA_VOL', '0'))",
        "'lambda_vol': LAMBDA_VOL",
        "'--lambda-vol', str(LAMBDA_VOL)",
        "'--vol-temperature', str(VOL_TEMPERATURE)",
        "'--vol-top-p', str(VOL_TOP_P)",
        "'--vol-samples', str(VOL_SAMPLES)",
        "'--lambda-path', '0' if LAMBDA_VOL else str(LAMBDA_PATH)",
        'tests/test_stage3_vol_alignment.py',
        "f'CE+{LAMBDA_VOL:g}*log_vol_huber_T{VOL_TEMPERATURE:g}_p{VOL_TOP_P:g}_N{VOL_SAMPLES}'",
    ):
        assert token in source, token


def test_trainer_refuses_c3_dashboard_and_requires_vol_exclusive():
    source = TRAINER.read_text()
    for token in (
        "p.add_argument('--lambda-vol', type=float, default=0.0)",
        "p.add_argument('--vol-temperature', type=float, default=0.65)",
        "p.add_argument('--vol-top-p', type=float, default=0.8)",
        "p.add_argument('--vol-samples', type=int, default=5)",
        "Refusing to reuse the T=1 top-16 vol-cal dashboard",
        "vol alignment requires lambda_path=0 and ce_rank disabled",
        "Refusing to reuse Stage3 C3 path-alignment dashboard for vol calibration",
        "os.environ['KRONOS_STAGE3_VOL_LOSS'] = '1' if a.lambda_vol else '0'",
        'vol_vs_ce_grad_norm_ratio',
        "f'token_ce+{a.lambda_vol:g}*log_vol_huber'",
    ):
        assert token in source, token
    assert "SWANLAB_RUN_ID', 'small_0.1_stage3_joint_path_alignment_from_c2_best_v2')" in source


def test_vol_loss_softmaxes_live_joint_logp_not_detached_decode_weights():
    source = (ROOT / 'finetune/stage3_vol_alignment.py').read_text()
    assert 'production_style_mixture_decode' in source
    assert 'temperature: float = 0.65' in source
    assert 'top_p: float = 0.8' in source
    assert 'candidates: int = 5' in source
    assert 'vol alignment mixture weights are detached' in source


def test_checkpoint_schema_splits_vol_from_c3_path():
    source = MODEL.read_text()
    assert "schema': 'stage3_vol_calibration_v2' if self.vol_config.weight else 'stage3_conditional_joint_causal_v2'" in source
    assert 'compute_vol_alignment_loss' in source
    assert "raise ValueError('vol alignment requires lambda_path=0')" in source
