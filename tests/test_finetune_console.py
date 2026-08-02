import numpy as np

from finetune.dataset import QlibDataset
from webui.finetune_manager import FineTuneManager


def make_dataset(total_samples=1_509_252, samples_per_segment=20_000, split='train'):
    dataset = QlibDataset.__new__(QlibDataset)
    dataset.data_type = split
    dataset.total_samples = total_samples
    dataset.n_samples = samples_per_segment
    dataset.coverage_order = np.random.default_rng(100).permutation(total_samples)
    dataset.active_positions = dataset.coverage_order[:samples_per_segment]
    dataset.coverage_start = 0
    return dataset


def test_default_segments_cover_every_training_window_without_replacement():
    dataset = make_dataset()
    covered = np.zeros(dataset.total_samples, dtype=bool)

    for segment in range(76):
        dataset.set_epoch_seed(segment)
        assert np.unique(dataset.active_positions).size == len(dataset.active_positions)
        assert not covered[dataset.active_positions].any()
        covered[dataset.active_positions] = True

    assert len(dataset.active_positions) == 9_252
    assert covered.all()


def test_validation_subset_is_fixed_across_segments():
    dataset = make_dataset(total_samples=10_000, samples_per_segment=2_000, split='val')
    expected = dataset.active_positions.copy()

    dataset.set_epoch_seed(19)

    np.testing.assert_array_equal(dataset.active_positions, expected)


def test_manager_default_estimate_requires_full_coverage(monkeypatch, tmp_path):
    manager = FineTuneManager(tmp_path)
    monkeypatch.setattr(manager, 'dataset_stats', lambda: {
        'train': {'symbols': 1176, 'rows': 1_626_852, 'windows': 1_509_252},
        'val': {'symbols': 1176, 'rows': 0, 'windows': 100_000},
        'window': 101,
    })

    estimate = manager.estimate(20_000, 2_000, 1, 5)

    assert estimate['segments_per_coverage'] == 76
    assert estimate['total_segments'] == 81


def test_manager_start_passes_batch_size_and_closes_log_on_failure(monkeypatch, tmp_path):
    manager = FineTuneManager(tmp_path)
    base_model = tmp_path / 'Kronos-base'
    base_model.mkdir()
    (base_model / 'model.safetensors').touch()
    (base_model / 'config.json').write_text('{}')
    monkeypatch.setattr(manager, 'estimate', lambda *args: {
        'total_segments': 6,
        'estimated_hours': 0.1,
    })

    captured = {}

    def fail_to_start(*args, **kwargs):
        captured.update(kwargs)
        raise OSError('launch failed')

    monkeypatch.setattr('webui.finetune_manager.subprocess.Popen', fail_to_start)
    payload = {
        'mode': 'discrete',
        'base_model': 'kronos-base-local',
        'output_name': 'coverage_test',
        'samples_per_segment': 800,
        'validation_samples': 200,
        'coverage_passes': 1,
        'batch_size': 8,
        'patience': 5,
    }

    try:
        manager.start(payload)
    except OSError as exc:
        assert str(exc) == 'launch failed'
    else:
        raise AssertionError('Expected subprocess startup failure')

    assert captured['env']['KRONOS_BATCH_SIZE'] == '8'
    assert captured['env']['KRONOS_PREDICTOR_PATH'] == str(base_model.resolve())
    assert manager.log_handle is None


def test_finetune_routes_render_and_return_configuration(monkeypatch):
    import webui.finetune_app as web_app

    stats = {
        'train': {'symbols': 1176, 'rows': 1_626_852, 'windows': 1_509_252},
        'val': {'symbols': 1176, 'rows': 0, 'windows': 100_000},
        'window': 101,
    }
    defaults = {
        'mode': 'discrete', 'base_model': 'kronos-base-local',
        'output_name': 'coverage_test',
        'samples_per_segment': 20_000, 'validation_samples': 2_000,
        'coverage_passes': 1, 'batch_size': 4, 'patience': 5,
    }
    monkeypatch.setattr(web_app.finetune_manager, 'dataset_stats', lambda: stats)
    monkeypatch.setattr(web_app.finetune_manager, 'defaults', lambda: defaults)
    monkeypatch.setattr(web_app.finetune_manager, 'base_models', lambda: [{
        'id': 'kronos-base-local',
        'name': 'NeoQuasar/Kronos-base',
        'source': '本地基础权重',
    }])
    monkeypatch.setattr(web_app.finetune_manager, 'estimate', lambda *args: {
        'segments_per_coverage': 76, 'total_segments': 81,
        'estimated_seconds': 104_400, 'estimated_hours': 29.0,
        'train_windows': 1_509_252,
    })
    monkeypatch.setattr(web_app.finetune_manager, 'status', lambda: {
        'status': 'idle', 'running': False, 'job': {}, 'progress': {},
        'log': '', 'can_resume': False,
    })
    monkeypatch.setattr(web_app.finetune_manager, 'checkpoints', lambda: [])
    client = web_app.app.test_client()

    page = client.get('/')
    config = client.get('/api/finetune/config')

    assert page.status_code == 200
    assert 'A股模型增训' in page.get_data(as_text=True)
    assert config.status_code == 200
    assert config.get_json()['estimate']['segments_per_coverage'] == 76
    assert config.get_json()['base_models'][0]['name'] == 'NeoQuasar/Kronos-base'
    assert 'path' not in config.get_json()['base_models'][0]


def test_base_model_choices_include_local_base_and_checkpoints(tmp_path):
    manager = FineTuneManager(tmp_path)
    base = tmp_path / 'Kronos-base'
    checkpoint = tmp_path / 'outputs' / 'models' / 'candidate' / 'checkpoints' / 'best_model'
    for directory in (base, checkpoint):
        directory.mkdir(parents=True)
        (directory / 'model.safetensors').touch()
        (directory / 'config.json').write_text('{}')

    choices = manager.base_models()

    assert choices == [
        {
            'id': 'kronos-base-local',
            'name': 'NeoQuasar/Kronos-base',
            'source': '本地基础权重',
        },
        {
            'id': 'checkpoint:candidate',
            'name': 'candidate',
            'source': '已训练 checkpoint',
        },
    ]
    assert all('path' not in choice for choice in choices)


def test_metrics_history_recovers_running_and_validation_loss(tmp_path):
    manager = FineTuneManager(tmp_path)
    log_path = tmp_path / 'outputs' / 'models' / 'run' / 'training.log'
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        '[Rank 0, Segment 1/81, Step 100/5000] LR 0.000001, Loss: 3.0271\n'
        '--- Coverage Segment 1/81 Summary ---\n'
        'Validation Loss: 2.9512\n'
    )

    metrics = manager._metrics_history(log_path)

    assert metrics['train'][0]['step'] == 100
    assert metrics['train'][0]['loss'] == 3.0271
    assert metrics['validation'][0]['segment'] == 1
    assert metrics['validation'][0]['best_loss'] == 2.9512


def test_stale_stopping_job_is_normalized_after_process_exit(tmp_path):
    manager = FineTuneManager(tmp_path)
    manager.job = {
        'status': 'stopping',
        'pid': 999_999_999,
        'output_name': 'stopped_run',
    }

    status = manager.status()

    assert status['status'] == 'stopped'
    assert status['running'] is False
