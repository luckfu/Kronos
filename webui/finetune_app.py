"""Standalone web application for A-share Kronos fine-tuning."""

import os
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

try:
    from .finetune_manager import FineTuneManager
except ImportError:
    from finetune_manager import FineTuneManager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)
CORS(app)
finetune_manager = FineTuneManager(PROJECT_ROOT)


def automatic_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


@app.route('/')
def index():
    return render_template('finetune.html')


@app.route('/api/finetune/config')
def get_finetune_config():
    try:
        defaults = finetune_manager.defaults()
        estimate = finetune_manager.estimate(
            defaults['samples_per_segment'],
            defaults['validation_samples'],
            defaults['coverage_passes'],
            defaults['patience'],
        )
        return jsonify({
            'success': True,
            'device': automatic_device(),
            'base_models': finetune_manager.base_models(),
            'dataset': finetune_manager.dataset_stats(),
            'defaults': defaults,
            'estimate': estimate,
            'status': finetune_manager.status(),
            'checkpoints': finetune_manager.checkpoints(),
        })
    except Exception as exc:
        return jsonify({'error': f'增训配置加载失败：{exc}'}), 500


@app.route('/api/finetune/estimate', methods=['POST'])
def estimate_finetune_job():
    try:
        data = request.get_json() or {}
        estimate = finetune_manager.estimate(
            int(data.get('samples_per_segment', 20_000)),
            int(data.get('validation_samples', 2_000)),
            int(data.get('coverage_passes', 1)),
            int(data.get('patience', 5)),
        )
        return jsonify({'success': True, 'estimate': estimate})
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'训练规模估算失败：{exc}'}), 500


@app.route('/api/finetune/status')
def get_finetune_status():
    return jsonify({
        'success': True,
        **finetune_manager.status(),
        'checkpoints': finetune_manager.checkpoints(),
    })


@app.route('/api/finetune/start', methods=['POST'])
def start_finetune_job():
    try:
        result = finetune_manager.start(request.get_json() or {})
        return jsonify({'success': True, **result})
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception as exc:
        return jsonify({'error': f'增训启动失败：{exc}'}), 500


@app.route('/api/finetune/stop', methods=['POST'])
def stop_finetune_job():
    try:
        return jsonify({'success': True, **finetune_manager.stop()})
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception as exc:
        return jsonify({'error': f'停止请求失败：{exc}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('KRONOS_FINETUNE_PORT', '7071'))
    print(f'Starting standalone Kronos fine-tuning app on port {port}...')
    app.run(host='0.0.0.0', port=port, debug=False)
