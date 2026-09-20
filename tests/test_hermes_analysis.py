import json
import subprocess

import pytest

from webui import app as web_app
from webui import hermes_analysis
from tests.test_daily_rankings import (
    forbid_live_name_lookups,
    published_run,
    reset_stock_name_cache,
)
from tests.test_web_auth import enable_auth


@pytest.fixture(autouse=True)
def reset_hermes_state(monkeypatch):
    hermes_analysis.clear_cache()
    monkeypatch.delenv('KRONOS_HERMES_DISABLED', raising=False)
    monkeypatch.delenv('KRONOS_HERMES_BIN', raising=False)
    monkeypatch.delenv('KRONOS_HERMES_PROVIDER', raising=False)
    monkeypatch.delenv('KRONOS_HERMES_MODEL', raising=False)
    monkeypatch.delenv('KRONOS_HERMES_TIMEOUT', raising=False)
    monkeypatch.delenv('KRONOS_HERMES_CACHE_TTL', raising=False)
    yield
    hermes_analysis.clear_cache()


def explode_ranking_chart_history(*args, **kwargs):
    raise AssertionError('Hermes analysis must not load ranking chart history')


def explode_modal(*args, **kwargs):
    raise AssertionError('Hermes analysis must not re-run Modal inference')


def prepare_rankings(monkeypatch, tmp_path, names=None):
    published_run(tmp_path, names=names)
    monkeypatch.setattr(web_app, 'DAILY_PREDICTION_ROOT', tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path, names=names or {})
    forbid_live_name_lookups(monkeypatch)
    monkeypatch.setattr(web_app, 'ranking_chart_history', explode_ranking_chart_history)
    if hasattr(web_app, 'predict_remote'):
        monkeypatch.setattr(web_app, 'predict_remote', explode_modal)
    return web_app.app.test_client()


def fake_oneshot(prompt):
    assert 'sz.000063' in prompt
    assert '2026-09-18' in prompt
    assert '+3.30%' in prompt or '3.30%' in prompt
    return {
        'analysis': '**中兴通讯** 10日中位路径偏强，但排名靠后，不宜当作确定收益。',
        'model': 'deepseek-v4-pro',
        'provider': 'deepseek',
        'elapsed_sec': 1.25,
        'cached': False,
    }


def test_prompt_includes_ranking_fields_and_is_sanitized():
    prompt = hermes_analysis.build_hermes_prompt({
        'asof': '2026-09-18',
        'code': 'sz.000063',
        'name': '中兴通讯<script>alert(1)</script>\x00',
        'close_asof': 30.0,
        'predicted_return_d1': 0.007,
        'predicted_return_d10': 0.033,
        'close_p50_d1': 30.2,
        'close_p50_d10': 31.0,
        'rank_d10': 11,
        'sector_label': 'C39计算机、通信和其他电子设备制造业',
        'size_percentile': 0.85,
    })

    assert 'sz.000063' in prompt
    assert '中兴通讯' in prompt
    assert '<script>' not in prompt
    assert '\x00' not in prompt
    assert '2026-09-18' in prompt
    assert '30.00' in prompt
    assert '+0.70%' in prompt
    assert '+3.30%' in prompt
    assert '#11' in prompt
    assert '计算机、通信和其他电子设备制造业' in prompt
    assert len(prompt) < hermes_analysis.MAX_PROMPT_CHARS


def test_hermes_command_forces_deepseek(monkeypatch):
    command = hermes_analysis.hermes_command(
        'PROMPT',
        bin_path='/data/miniconda3/bin/hermes',
    )

    assert command[0] == '/data/miniconda3/bin/hermes'
    assert command[1:4] == ['chat', '-q', 'PROMPT']
    assert '--oneshot' in command
    assert '-Q' in command
    assert command[command.index('-m') + 1] == 'deepseek-v4-pro'
    assert command[command.index('--provider') + 1] == 'deepseek'


def test_run_hermes_oneshot_returns_stdout_only(monkeypatch):
    monkeypatch.setattr(hermes_analysis, 'resolve_hermes_bin', lambda: '/tmp/fake-hermes')
    recorded = {}

    def fake_run(command, **kwargs):
        recorded['command'] = command
        recorded['kwargs'] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='  这是分析正文  \n',
            stderr='DEEPSEEK_API_KEY=sk-secret\n',
        )

    monkeypatch.setattr(hermes_analysis.subprocess, 'run', fake_run)

    result = hermes_analysis.run_hermes_oneshot('简短提示')

    assert recorded['command'][0] == '/tmp/fake-hermes'
    assert recorded['kwargs']['capture_output'] is True
    assert recorded['kwargs'].get('shell') not in (True,)
    assert recorded['kwargs']['stdin'] is subprocess.DEVNULL
    assert result['analysis'] == '这是分析正文'
    assert result['provider'] == 'deepseek'
    assert result['model'] == 'deepseek-v4-pro'
    assert 'elapsed_sec' in result
    dumped = json.dumps(result)
    assert 'sk-secret' not in dumped
    assert 'DEEPSEEK_API_KEY' not in dumped


def test_run_hermes_oneshot_timeout(monkeypatch):
    monkeypatch.setattr(hermes_analysis, 'resolve_hermes_bin', lambda: '/tmp/fake-hermes')
    monkeypatch.setenv('KRONOS_HERMES_TIMEOUT', '12')

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get('timeout'))

    monkeypatch.setattr(hermes_analysis.subprocess, 'run', fake_run)

    with pytest.raises(hermes_analysis.HermesTimeoutError) as exc:
        hermes_analysis.run_hermes_oneshot('提示')
    assert '12' in str(exc.value)
    assert exc.value.status_code == 504


def test_missing_hermes_binary_is_unavailable(monkeypatch, tmp_path):
    missing = tmp_path / 'no-such-hermes'
    monkeypatch.setenv('KRONOS_HERMES_BIN', str(missing))
    with pytest.raises(hermes_analysis.HermesUnavailableError) as exc:
        hermes_analysis.resolve_hermes_bin()
    assert '未找到可执行文件' in str(exc.value)
    assert exc.value.status_code == 503


def test_hermes_analysis_endpoint_success(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path, names={'sz.000063': '中兴通讯'})
    monkeypatch.setattr(web_app.hermes_analysis, 'run_hermes_oneshot', fake_oneshot)
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/2026-09-18/000063/hermes-analysis')
    prefixed = client.post('/api/daily-rankings/2026-09-18/sz.000063/hermes-analysis')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['code'] == 'sz.000063'
    assert payload['name'] == '中兴通讯'
    assert payload['provider'] == 'deepseek'
    assert payload['model'] == 'deepseek-v4-pro'
    assert '中兴通讯' in payload['analysis']
    assert payload['cached'] is False
    assert prefixed.status_code == 200
    assert prefixed.get_json()['cached'] is True
    dumped = json.dumps(payload)
    assert 'sk-' not in dumped
    assert 'API_KEY' not in dumped


def test_hermes_analysis_cache_skips_second_subprocess(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path, names={'sz.000063': '中兴通讯'})
    calls = {'count': 0}

    def counting_oneshot(prompt):
        calls['count'] += 1
        return fake_oneshot(prompt)

    monkeypatch.setattr(web_app.hermes_analysis, 'run_hermes_oneshot', counting_oneshot)
    client = web_app.app.test_client()

    first = client.post('/api/daily-rankings/2026-09-18/000063/hermes-analysis')
    second = client.post('/api/daily-rankings/2026-09-18/sz.000063/hermes-analysis')

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls['count'] == 1
    assert first.get_json()['cached'] is False
    assert second.get_json()['cached'] is True
    assert second.get_json()['analysis'] == first.get_json()['analysis']


def test_hermes_analysis_missing_binary_returns_503(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path)
    monkeypatch.setenv('KRONOS_HERMES_BIN', str(tmp_path / 'missing-hermes'))
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/2026-09-18/000063/hermes-analysis')

    assert response.status_code == 503
    assert 'Hermes 命令不可用' in response.get_json()['error']


def test_hermes_analysis_disabled_returns_503(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path)
    monkeypatch.setenv('KRONOS_HERMES_DISABLED', '1')
    monkeypatch.setattr(
        web_app.hermes_analysis,
        'run_hermes_oneshot',
        lambda prompt: (_ for _ in ()).throw(AssertionError('disabled should not run')),
    )
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/2026-09-18/000063/hermes-analysis')

    assert response.status_code == 503
    assert '已禁用' in response.get_json()['error']


def test_hermes_analysis_timeout_returns_504(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path)

    def boom(prompt):
        raise hermes_analysis.HermesTimeoutError('Hermes 分析超时（120秒）。')

    monkeypatch.setattr(web_app.hermes_analysis, 'run_hermes_oneshot', boom)
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/2026-09-18/000063/hermes-analysis')

    assert response.status_code == 504
    assert '超时' in response.get_json()['error']


def test_hermes_analysis_unknown_symbol_returns_404(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path)
    monkeypatch.setattr(web_app.hermes_analysis, 'run_hermes_oneshot', fake_oneshot)
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/2026-09-18/600519/hermes-analysis')

    assert response.status_code == 404
    assert '未进入' in response.get_json()['error']
    assert '600519' in response.get_json()['error']


def test_hermes_analysis_invalid_asof_returns_400(monkeypatch, tmp_path):
    prepare_rankings(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/not-a-date/000063/hermes-analysis')

    assert response.status_code == 400


def test_hermes_analysis_requires_login(monkeypatch, tmp_path):
    enable_auth(monkeypatch, tmp_path)
    client = web_app.app.test_client()

    response = client.post('/api/daily-rankings/2026-09-18/000063/hermes-analysis')

    assert response.status_code == 401
    assert response.get_json()['error'] == 'authentication required'
    monkeypatch.setenv('KRONOS_AUTH_DISABLED', '1')
    monkeypatch.delenv('KRONOS_HTPASSWD_PATH', raising=False)
    web_app.configure_auth(web_app.app)


def test_daily_rankings_page_has_on_demand_hermes_button():
    page = web_app.app.test_client().get('/').get_data(as_text=True)
    open_detail = page[page.index('async function openDetail'):page.index('function shortDate')]

    assert 'Hermes 分析' in page
    assert 'hermes-analysis' in page
    assert "function analyzeWithHermes" in page
    assert "method: 'POST'" in page
    assert 'hermes-analysis' not in open_detail
    assert 'analyzeWithHermes(' not in open_detail
    assert 'resetHermesPanel()' in open_detail
    assert '历史行情 + 未来10日预测' in page
    assert 'drawChart(result)' in page


def test_deploy_and_service_document_hermes_cli():
    deploy = open('deploy/oracle-kronos/deploy.sh', encoding='utf-8').read()
    service = open('deploy/oracle-kronos/kronos-web.service', encoding='utf-8').read()
    readme = open('deploy/oracle-kronos/README.md', encoding='utf-8').read()

    assert 'webui/hermes_analysis.py' in deploy
    assert 'KRONOS_HERMES_BIN=/data/miniconda3/bin/hermes' in service
    assert 'KRONOS_HERMES_PROVIDER=deepseek' in service
    assert 'KRONOS_HERMES_MODEL=deepseek-v4-pro' in service
    assert 'Hermes 分析' in readme
    assert '~/.hermes/.env' in readme
    assert 'opc' in readme
