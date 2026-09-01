import ast
import math
from pathlib import Path


TRAINER = (
    Path(__file__).parents[1]
    / "finetune/kaggle_v1_beta_minimal/Kronos/finetune/train_predictor.py"
)


def load_optimizer_helpers():
    tree = ast.parse(TRAINER.read_text())
    wanted = {
        "segment_sample_count",
        "optimizer_steps_for_completed_segments",
        "is_condition_parameter",
        "parameter_uses_weight_decay",
        "warmup_cosine_multiplier",
        "two_speed_multiplier",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), namespace)
    return namespace


def test_optimizer_step_position_accounts_for_short_tail_segments():
    helper = load_optimizer_helpers()["optimizer_steps_for_completed_segments"]

    class Dataset:
        total_samples = 45_000
        n_samples = 20_000

    # One pass is 625 + 625 + 157 optimizer steps. Segment four starts pass two.
    assert helper(Dataset(), 3, world_size=1, batch_size=32) == 1_407
    assert helper(Dataset(), 4, world_size=1, batch_size=32) == 2_032


def test_warmup_cosine_hits_start_peak_and_minimum_for_both_families():
    schedule = load_optimizer_helpers()["warmup_cosine_multiplier"]
    total_steps = 1_320_510
    warmup_steps = 26_410

    assert math.isclose(
        1e-5 * schedule(0, total_steps, warmup_steps, 1e-6, 1e-5, 1e-6),
        1e-6,
    )
    assert math.isclose(
        1e-4 * schedule(0, total_steps, warmup_steps, 1e-5, 1e-4, 1e-6),
        1e-5,
    )
    assert math.isclose(
        schedule(warmup_steps, total_steps, warmup_steps, 1e-6, 1e-5, 1e-6),
        1.0,
    )
    assert math.isclose(
        1e-5 * schedule(total_steps, total_steps, warmup_steps, 1e-6, 1e-5, 1e-6),
        1e-6,
    )
    assert math.isclose(
        1e-4 * schedule(total_steps, total_steps, warmup_steps, 1e-5, 1e-4, 1e-6),
        1e-6,
    )


def test_embedding_norm_and_bias_parameters_do_not_decay():
    helpers = load_optimizer_helpers()
    uses_decay = helpers["parameter_uses_weight_decay"]

    class Parameter:
        def __init__(self, ndim):
            self.ndim = ndim

    assert not uses_decay("sector_emb.weight", Parameter(2))
    assert not uses_decay("transformer.10.norm1.weight", Parameter(1))
    assert not uses_decay("head.s1_head.bias", Parameter(1))
    assert uses_decay("transformer.10.attn.q_proj.weight", Parameter(2))
    assert uses_decay("size_mlp.0.weight", Parameter(2))


def test_condition_parameter_names_are_isolated_from_adaptation_parameters():
    is_condition = load_optimizer_helpers()["is_condition_parameter"]

    assert is_condition("sector_emb.weight")
    assert is_condition("size_mlp.2.weight")
    assert is_condition("size_path_mlp.2.weight")
    assert not is_condition("transformer.10.attn.q_proj.weight")


def test_two_speed_condition_schedule_is_continuous_and_monotonic():
    schedule = load_optimizer_helpers()["two_speed_multiplier"]
    total_steps = 1_320_510
    warmup_steps = round(total_steps * 0.005)
    fast_decay_steps = round(total_steps * 0.075)

    condition_lrs = [
        3e-5 * schedule(
            step, total_steps, warmup_steps,
            1e-5, 3e-5, 1e-6, "condition",
            fast_decay_steps, 1e-5,
        )
        for step in (
            0, warmup_steps, warmup_steps + 1,
            fast_decay_steps, fast_decay_steps + 1, total_steps,
        )
    ]

    assert math.isclose(condition_lrs[0], 1e-5)
    assert math.isclose(condition_lrs[1], 3e-5)
    assert condition_lrs[1] >= condition_lrs[2] >= condition_lrs[3]
    assert math.isclose(condition_lrs[3], 1e-5)
    assert condition_lrs[3] >= condition_lrs[4] >= condition_lrs[5]
    assert math.isclose(condition_lrs[-1], 1e-6)


def test_two_speed_adaptation_uses_global_conservative_cosine():
    schedule = load_optimizer_helpers()["two_speed_multiplier"]
    total_steps = 1_320_510
    warmup_steps = round(total_steps * 0.005)
    fast_decay_steps = round(total_steps * 0.075)

    assert math.isclose(
        3e-6 * schedule(
            0, total_steps, warmup_steps, 1e-6, 3e-6, 5e-7,
            "adaptation", fast_decay_steps, 1e-5,
        ),
        1e-6,
    )
    assert math.isclose(
        3e-6 * schedule(
            warmup_steps, total_steps, warmup_steps, 1e-6, 3e-6, 5e-7,
            "adaptation", fast_decay_steps, 1e-5,
        ),
        3e-6,
    )
    assert math.isclose(
        3e-6 * schedule(
            total_steps, total_steps, warmup_steps, 1e-6, 3e-6, 5e-7,
            "adaptation", fast_decay_steps, 1e-5,
        ),
        5e-7,
    )


def test_a800_amp_supports_bfloat16_without_gradient_scaling():
    trainer = TRAINER.read_text()
    config = TRAINER.parent.joinpath("config.py").read_text()
    launcher = TRAINER.parents[3].joinpath(
        "run_a800_v1_beta_natural_twospeed_v2.sh"
    ).read_text()

    assert "KRONOS_AMP_DTYPE" in config
    assert "return torch.bfloat16" in trainer
    assert "scale_gradients = amp_dtype == torch.float16" in trainer
    assert "amp_dtype=amp_dtype_name" in trainer
    assert "KRONOS_AMP_DTYPE=bfloat16" in launcher
