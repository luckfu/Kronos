"""Local process manager for long-running Kronos fine-tuning jobs."""

import json
import math
import os
import pickle
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


class FineTuneManager:
    OUTPUT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{2,79}$')

    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.dataset_dir = self.project_root / 'data' / 'a_share' / 'processed_datasets'
        self.models_dir = self.project_root / 'outputs' / 'models'
        self.jobs_dir = self.project_root / 'outputs' / 'finetune_jobs'
        self.state_path = self.jobs_dir / 'current_job.json'
        self.lock = threading.RLock()
        self.process = None
        self.log_handle = None
        self.job = self._read_json(self.state_path) or {}
        self._stats_cache = None
        self._stats_signature = None

    @staticmethod
    def _read_json(path):
        try:
            with open(path) as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _write_state(self):
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix('.json.tmp')
        with open(temporary, 'w') as handle:
            json.dump(self.job, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, self.state_path)

    def _base_model_registry(self):
        configured = os.getenv('KRONOS_PREDICTOR_PATH', '').strip()
        candidates = [
            ('NeoQuasar/Kronos-base', '本地基础权重', Path(configured).expanduser())
            if configured else None,
            self.project_root / 'Kronos-base',
            self.project_root.parent / 'Kronos-base',
        ]
        registry = {}
        seen_paths = set()
        local_index = 0
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, tuple):
                name, source, path = candidate
            else:
                name, source, path = 'NeoQuasar/Kronos-base', '本地基础权重', candidate
            path = path.resolve()
            if path in seen_paths:
                continue
            if not (path / 'model.safetensors').exists() or not (path / 'config.json').exists():
                continue
            model_id = 'kronos-base-local' if local_index == 0 else f'kronos-base-local-{local_index + 1}'
            registry[model_id] = {
                'id': model_id,
                'name': name,
                'source': source,
                'path': path,
            }
            seen_paths.add(path)
            local_index += 1

        if self.models_dir.exists():
            for directory in sorted(self.models_dir.iterdir(), key=lambda item: item.name):
                checkpoint = directory / 'checkpoints' / 'best_model'
                if not (checkpoint / 'model.safetensors').exists() or not (checkpoint / 'config.json').exists():
                    continue
                resolved = checkpoint.resolve()
                if resolved in seen_paths:
                    continue
                model_id = f'checkpoint:{directory.name}'
                registry[model_id] = {
                    'id': model_id,
                    'name': directory.name,
                    'source': '已训练 checkpoint',
                    'path': resolved,
                }
                seen_paths.add(resolved)
        return registry

    def base_models(self):
        return [
            {key: item[key] for key in ('id', 'name', 'source')}
            for item in self._base_model_registry().values()
        ]

    def _resolve_base_model(self, model_id=None):
        registry = self._base_model_registry()
        if not registry:
            raise FileNotFoundError(
                '没有找到可用基础模型，请通过 KRONOS_PREDICTOR_PATH 配置权重目录'
            )
        selected_id = str(model_id or next(iter(registry))).strip()
        if selected_id not in registry:
            raise ValueError('所选基础模型不存在或权重不完整')
        return registry[selected_id]

    def dataset_stats(self):
        paths = {
            split: self.dataset_dir / f'{split}_data.pkl'
            for split in ('train', 'val')
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f'训练数据不存在：{missing[0]}')
        signature = tuple((str(path), path.stat().st_mtime_ns) for path in paths.values())
        if self._stats_cache is not None and signature == self._stats_signature:
            return self._stats_cache

        result = {}
        window = 101
        for split, path in paths.items():
            with open(path, 'rb') as handle:
                panel = pickle.load(handle)
            rows = sum(len(frame) for frame in panel.values())
            windows = sum(max(len(frame) - window + 1, 0) for frame in panel.values())
            result[split] = {
                'symbols': len(panel),
                'rows': rows,
                'windows': windows,
            }
        result['window'] = window
        result['measured_seconds_per_train_window'] = 0.0625
        self._stats_cache = result
        self._stats_signature = signature
        return result

    def defaults(self):
        stats = self.dataset_stats()
        base_model = self._resolve_base_model()
        samples_per_segment = 20_000
        segments = math.ceil(stats['train']['windows'] / samples_per_segment)
        return {
            'mode': 'discrete',
            'base_model': base_model['id'],
            'output_name': 'a_share_size_full_coverage_v1',
            'samples_per_segment': samples_per_segment,
            'validation_samples': 2_000,
            'coverage_passes': 1,
            'batch_size': 4,
            'patience': 5,
            'requested_segments': segments + 5,
            'predictor_learning_rate': 1e-5,
            'condition_learning_rate': 1e-3,
        }

    def estimate(self, samples_per_segment, validation_samples, coverage_passes, patience):
        stats = self.dataset_stats()
        train_windows = stats['train']['windows']
        samples_per_segment = train_windows if samples_per_segment <= 0 else min(
            samples_per_segment, train_windows
        )
        segments_per_pass = math.ceil(train_windows / samples_per_segment)
        total_segments = segments_per_pass * coverage_passes + patience
        coverage_windows = train_windows * coverage_passes
        patience_windows = patience * samples_per_segment
        training_seconds = (coverage_windows + patience_windows) * 0.0625
        validation_seconds = total_segments * validation_samples * 0.018
        return {
            'segments_per_coverage': segments_per_pass,
            'total_segments': total_segments,
            'estimated_seconds': int(training_seconds + validation_seconds),
            'estimated_hours': round((training_seconds + validation_seconds) / 3600, 1),
            'train_windows': train_windows,
        }

    def _is_pid_running(self, pid):
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False

    def is_running(self):
        if self.process is not None:
            return self.process.poll() is None
        return self._is_pid_running(self.job.get('pid')) and self.job.get('status') in {
            'running', 'stopping'
        }

    def _tail_log(self, path, limit_bytes=64 * 1024):
        try:
            with open(path, 'rb') as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit_bytes))
                return handle.read().decode('utf-8', errors='replace')[-20_000:]
        except OSError:
            return ''

    def _metrics_history(self, log_path):
        if not log_path:
            return {'train': [], 'validation': []}
        log_path = Path(log_path)
        points = {}
        metrics_path = log_path.parent / 'metrics.jsonl'
        try:
            with open(metrics_path) as handle:
                for line in handle:
                    try:
                        point = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metric_type = point.get('type')
                    if metric_type not in {'train', 'validation'}:
                        continue
                    key = (metric_type, int(point.get('segment', 0)), int(point.get('step', 0)))
                    points[key] = point
        except OSError:
            pass

        try:
            log_text = log_path.read_text(errors='replace')
        except OSError:
            log_text = ''
        train_pattern = re.compile(
            r'\[Rank \d+, Segment (\d+)/(\d+), Step (\d+)/(\d+)\] '
            r'LR ([0-9.eE+-]+), Loss: ([0-9.]+)'
        )
        for match in train_pattern.finditer(log_text):
            segment, total_segments, step, total_steps, learning_rate, loss = match.groups()
            key = ('train', int(segment), int(step))
            points.setdefault(key, {
                'type': 'train',
                'segment': int(segment),
                'total_segments': int(total_segments),
                'step': int(step),
                'total_steps': int(total_steps),
                'loss': float(loss),
                'learning_rate': float(learning_rate),
            })

        validation_pattern = re.compile(
            r'--- Coverage Segment (\d+)/(\d+) Summary ---\s*'
            r'Validation Loss: ([0-9.]+)'
        )
        best_loss = float('inf')
        for match in validation_pattern.finditer(log_text):
            segment, total_segments, loss = match.groups()
            loss = float(loss)
            best_loss = min(best_loss, loss)
            key = ('validation', int(segment), 0)
            points.setdefault(key, {
                'type': 'validation',
                'segment': int(segment),
                'total_segments': int(total_segments),
                'step': 0,
                'loss': loss,
                'best_loss': best_loss,
            })

        train = sorted(
            (point for point in points.values() if point['type'] == 'train'),
            key=lambda point: (point['segment'], point.get('step', 0)),
        )[-5000:]
        validation = sorted(
            (point for point in points.values() if point['type'] == 'validation'),
            key=lambda point: point['segment'],
        )
        return {'train': train, 'validation': validation}

    def status(self):
        with self.lock:
            if self.process is not None and self.process.poll() is not None:
                return_code = self.process.returncode
                self.process = None
                if self.log_handle is not None:
                    self.log_handle.close()
                    self.log_handle = None
                progress = self._read_json(Path(self.job.get('progress_path', ''))) or {}
                self.job['status'] = progress.get(
                    'status', 'completed' if return_code == 0 else 'failed'
                )
                self.job['return_code'] = return_code
                self.job['finished_at'] = self._utc_now()
                self._write_state()

            progress = self._read_json(Path(self.job.get('progress_path', ''))) or {}
            running = self.is_running()
            status = self.job.get('status', 'idle')
            if not running and status in {'running', 'stopping'}:
                status = 'stopped' if status == 'stopping' else 'failed'
                self.job['status'] = status
                self.job['finished_at'] = self.job.get('finished_at') or self._utc_now()
                self._write_state()
            if running and status not in {'stopping'}:
                status = 'running'
            return {
                'status': status,
                'running': running,
                'job': {
                    key: self.job.get(key)
                    for key in (
                        'output_name', 'mode', 'pid', 'started_at', 'finished_at',
                        'base_model', 'resume', 'return_code', 'estimated_hours',
                    )
                },
                'progress': progress,
                'metrics': self._metrics_history(self.job.get('log_path')),
                'log': self._tail_log(self.job.get('log_path', '')),
                'can_resume': bool(
                    self.job.get('resume_path')
                    and Path(self.job['resume_path']).exists()
                    and not running
                ),
            }

    def start(self, payload):
        with self.lock:
            if self.is_running():
                raise RuntimeError('已有增训任务正在运行')

            mode = str(payload.get('mode', 'discrete')).strip().lower()
            if mode not in {'discrete', 'hybrid'}:
                raise ValueError('训练模式必须是 discrete 或 hybrid')
            output_name = str(payload.get('output_name', '')).strip()
            if not self.OUTPUT_NAME_PATTERN.fullmatch(output_name):
                raise ValueError('输出名称只能包含字母、数字、下划线和短横线，长度 3–80')

            samples_per_segment = int(payload.get('samples_per_segment', 20_000))
            validation_samples = int(payload.get('validation_samples', 2_000))
            coverage_passes = int(payload.get('coverage_passes', 1))
            batch_size = int(payload.get('batch_size', 4))
            patience = int(payload.get('patience', 5))
            resume = bool(payload.get('resume', False))
            base_model = self._resolve_base_model(payload.get('base_model'))
            if not 800 <= samples_per_segment <= 200_000:
                raise ValueError('每段训练窗口必须在 800–200000 之间')
            if not 200 <= validation_samples <= 10_000:
                raise ValueError('验证窗口必须在 200–10000 之间')
            if not 1 <= coverage_passes <= 3:
                raise ValueError('完整覆盖遍数必须在 1–3 之间')
            if not 1 <= batch_size <= 16:
                raise ValueError('批大小必须在 1–16 之间')
            if not 0 <= patience <= 20:
                raise ValueError('早停耐心必须在 0–20 之间')

            estimate = self.estimate(
                samples_per_segment, validation_samples, coverage_passes, patience
            )
            output_dir = self.models_dir / output_name
            resume_path = output_dir / 'checkpoints' / 'last_state.pt'
            if resume and not resume_path.exists():
                raise FileNotFoundError('该输出目录没有可恢复的训练状态')
            best_model_path = output_dir / 'checkpoints' / 'best_model' / 'model.safetensors'
            if not resume and (resume_path.exists() or best_model_path.exists()):
                raise FileExistsError('输出目录已有 checkpoint，请更换名称或选择恢复训练')

            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / 'training.log'
            progress_path = output_dir / 'progress.json'
            log_mode = 'ab' if resume else 'wb'
            self.log_handle = open(log_path, log_mode, buffering=0)

            env = os.environ.copy()
            env.update({
                'KMP_DUPLICATE_LIB_OK': 'TRUE',
                'PYTHONPATH': f'{self.project_root}:{self.project_root / "finetune"}',
                'KRONOS_USE_SIZE_PERCENTILE': '1' if mode == 'hybrid' else '0',
                'KRONOS_EPOCHS': str(estimate['total_segments']),
                'KRONOS_EARLY_STOPPING_PATIENCE': str(patience),
                'KRONOS_PREDICTOR_PATH': str(base_model['path']),
                'KRONOS_PREDICTOR_SAVE_FOLDER': output_name,
                'KRONOS_TRAIN_SAMPLES_PER_SEGMENT': str(samples_per_segment),
                'KRONOS_VALIDATION_SAMPLES': str(validation_samples),
                'KRONOS_COVERAGE_PASSES': str(coverage_passes),
                'KRONOS_BATCH_SIZE': str(batch_size),
                'KRONOS_REQUIRE_FULL_COVERAGE': '1',
                'KRONOS_RESUME_TRAINING': '1' if resume else '0',
            })
            command = [sys.executable, '-u', 'finetune/train_predictor.py']
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdout=self.log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                self.log_handle.close()
                self.log_handle = None
                raise
            self.job = {
                'status': 'running',
                'output_name': output_name,
                'mode': mode,
                'base_model': {
                    key: base_model[key] for key in ('id', 'name', 'source')
                },
                'pid': self.process.pid,
                'started_at': self._utc_now(),
                'finished_at': None,
                'resume': resume,
                'return_code': None,
                'estimated_hours': estimate['estimated_hours'],
                'log_path': str(log_path),
                'progress_path': str(progress_path),
                'resume_path': str(resume_path),
            }
            self._write_state()
            return self.status()

    def stop(self):
        with self.lock:
            if not self.is_running():
                raise RuntimeError('当前没有运行中的增训任务')
            pid = int(self.job['pid'])
            os.killpg(os.getpgid(pid), signal.SIGINT)
            self.job['status'] = 'stopping'
            self._write_state()
            return self.status()

    def checkpoints(self):
        result = []
        if not self.models_dir.exists():
            return result
        for directory in self.models_dir.iterdir():
            if not directory.is_dir():
                continue
            summary = self._read_json(directory / 'summary.json') or {}
            progress = self._read_json(directory / 'progress.json') or {}
            best_model = directory / 'checkpoints' / 'best_model' / 'model.safetensors'
            resume_state = directory / 'checkpoints' / 'last_state.pt'
            if not any((summary, progress, best_model.exists(), resume_state.exists())):
                continue
            result.append({
                'name': directory.name,
                'status': (
                    self.job.get('status')
                    if directory.name == self.job.get('output_name')
                    and not self.is_running()
                    else progress.get('status', 'completed' if best_model.exists() else 'unknown')
                ),
                'best_val_loss': progress.get(
                    'best_val_loss', summary.get('final_result', {}).get('best_val_loss')
                ),
                'current_segment': progress.get('current_segment'),
                'total_segments': progress.get('total_segments'),
                'has_best_model': best_model.exists(),
                'can_resume': resume_state.exists(),
                'updated_at': progress.get('updated_at'),
            })
        return sorted(result, key=lambda item: item.get('updated_at') or '', reverse=True)[:20]
