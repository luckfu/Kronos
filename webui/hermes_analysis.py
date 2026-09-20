"""On-demand Hermes/DeepSeek analysis for daily ranking detail cards.

Kronos never scrapes the Hermes WebUI or reads ~/.hermes/.env. It execs the
local CLI as the same service user (opc on Oracle) and returns stdout only.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import threading
import time

CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
HTML_TAGS = re.compile(r'<[^>]*>')
LEADING_SECTOR_CODE = re.compile(r'^[A-Z]\d+')
DEFAULT_BIN = 'hermes'
DEFAULT_PROVIDER = 'deepseek'
DEFAULT_MODEL = 'deepseek-v4-pro'
DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_CACHE_TTL_SEC = 1800
MAX_TIMEOUT_SEC = 180.0
MAX_PROMPT_CHARS = 2500
MAX_ANALYSIS_CHARS = 12000
MAX_CACHE_ENTRIES = 128
FIELD_LIMITS = {
    'code': 16,
    'name': 40,
    'asof': 16,
    'sector_label': 80,
}


class HermesError(Exception):
    status_code = 500


class HermesDisabledError(HermesError):
    status_code = 503


class HermesUnavailableError(HermesError):
    status_code = 503


class HermesTimeoutError(HermesError):
    status_code = 504


class HermesFailedError(HermesError):
    status_code = 502


_cache = {}
_cache_lock = threading.Lock()


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def hermes_disabled():
    return _env_flag('KRONOS_HERMES_DISABLED')


def hermes_bin():
    return (os.getenv('KRONOS_HERMES_BIN') or DEFAULT_BIN).strip() or DEFAULT_BIN


def hermes_provider():
    return (os.getenv('KRONOS_HERMES_PROVIDER') or DEFAULT_PROVIDER).strip() or DEFAULT_PROVIDER


def hermes_model():
    return (os.getenv('KRONOS_HERMES_MODEL') or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def hermes_timeout():
    raw = os.getenv('KRONOS_HERMES_TIMEOUT', str(int(DEFAULT_TIMEOUT_SEC)))
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SEC
    return max(10.0, min(MAX_TIMEOUT_SEC, timeout))


def hermes_cache_ttl():
    raw = os.getenv('KRONOS_HERMES_CACHE_TTL', str(DEFAULT_CACHE_TTL_SEC))
    try:
        ttl = int(raw)
    except (TypeError, ValueError):
        ttl = DEFAULT_CACHE_TTL_SEC
    return max(0, ttl)


def sanitize_text(value, max_len=80):
    text = HTML_TAGS.sub('', str(value or ''))
    text = CONTROL_CHARS.sub('', text).replace('\r', ' ').replace('\n', ' ')
    text = ' '.join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def format_price(value):
    number = _finite_number(value)
    return '—' if number is None else f'{number:.2f}'


def format_pct(value, signed=True):
    number = _finite_number(value)
    if number is None:
        return '—'
    sign = '+' if signed and number >= 0 else ''
    return f'{sign}{number * 100:.2f}%'


def format_rank(value):
    number = _finite_number(value)
    if number is None:
        return '—'
    return str(int(number))


def strip_sector_label(value):
    text = sanitize_text(value, FIELD_LIMITS['sector_label'])
    stripped = LEADING_SECTOR_CODE.sub('', text).strip()
    return stripped or text or '—'


def resolve_hermes_bin():
    configured = hermes_bin()
    if os.path.dirname(configured):
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        raise HermesUnavailableError(
            f'Hermes 命令不可用：未找到可执行文件 {configured}。'
            '请将 hermes 加入 opc 的 PATH，或设置 KRONOS_HERMES_BIN。'
        )
    found = shutil.which(configured)
    if found:
        return found
    raise HermesUnavailableError(
        f'Hermes 命令不可用：PATH 中没有 {configured}。'
        '请将 hermes 加入 opc 的 PATH，或设置 KRONOS_HERMES_BIN。'
    )


def hermes_command(prompt, bin_path=None, model=None, provider=None):
    return [
        bin_path or resolve_hermes_bin(),
        'chat',
        '-q', prompt,
        '--oneshot',
        '-Q',
        '-m', model or hermes_model(),
        '--provider', provider or hermes_provider(),
    ]


def build_hermes_prompt(facts):
    facts = facts or {}
    code = sanitize_text(facts.get('code'), FIELD_LIMITS['code']) or '未知代码'
    name = sanitize_text(facts.get('name'), FIELD_LIMITS['name']) or '未知名称'
    asof = sanitize_text(facts.get('asof'), FIELD_LIMITS['asof']) or '未知日期'
    sector = strip_sector_label(facts.get('sector_label'))
    prompt = (
        '你是严谨的A股研究助手。根据下列已发布的Kronos截面预测，用简体中文写一段不超过400字的分析。'
        '只使用给定数字，不要编造财报、新闻或未提供的行情。明确这是模型预测而非投资建议。\n\n'
        f'股票代码：{code}\n'
        f'名称：{name}\n'
        f'信号日：{asof}\n'
        f'行业：{sector}\n'
        f'信号日收盘：{format_price(facts.get("close_asof"))}\n'
        f'市值百分位：{format_pct(facts.get("size_percentile"), signed=False)}\n'
        f'全市场10日预测排名：#{format_rank(facts.get("rank_d10"))}\n'
        f'1日中位预测收益：{format_pct(facts.get("predicted_return_d1"))}\n'
        f'10日中位预测收益：{format_pct(facts.get("predicted_return_d10"))}\n'
        f'1日中位预测收盘：{format_price(facts.get("close_p50_d1"))}\n'
        f'10日中位预测收盘：{format_price(facts.get("close_p50_d10"))}\n\n'
        '请覆盖：1) 这些预测数字意味着什么；2) 主要风险与不确定性；3) 如何理解排名和概率区间。'
        '不要给出具体买卖点或仓位建议。'
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS].rstrip()
    return prompt


def cache_key(asof, code):
    return (str(asof), str(code), hermes_provider(), hermes_model())


def clear_cache():
    with _cache_lock:
        _cache.clear()


def get_cached(asof, code):
    ttl = hermes_cache_ttl()
    if ttl <= 0:
        return None
    key = cache_key(asof, code)
    now = time.monotonic()
    with _cache_lock:
        item = _cache.get(key)
        if not item:
            return None
        expires_at, payload = item
        if expires_at <= now:
            _cache.pop(key, None)
            return None
        cached = dict(payload)
        cached['cached'] = True
        return cached


def put_cached(asof, code, payload):
    ttl = hermes_cache_ttl()
    if ttl <= 0:
        return
    key = cache_key(asof, code)
    stored = dict(payload)
    stored['cached'] = False
    expires_at = time.monotonic() + ttl
    with _cache_lock:
        if len(_cache) >= MAX_CACHE_ENTRIES:
            now = time.monotonic()
            for stale_key in [item for item, (exp, _) in _cache.items() if exp <= now]:
                _cache.pop(stale_key, None)
            while len(_cache) >= MAX_CACHE_ENTRIES:
                _cache.pop(next(iter(_cache)))
        _cache[key] = (expires_at, stored)


def run_hermes_oneshot(prompt):
    if hermes_disabled():
        raise HermesDisabledError('Hermes 分析已禁用（KRONOS_HERMES_DISABLED）。')
    cleaned = CONTROL_CHARS.sub('', str(prompt or '')).strip()
    if not cleaned:
        raise HermesFailedError('分析提示为空。')
    if len(cleaned) > MAX_PROMPT_CHARS:
        cleaned = cleaned[:MAX_PROMPT_CHARS].rstrip()
    bin_path = resolve_hermes_bin()
    timeout = hermes_timeout()
    command = hermes_command(cleaned, bin_path=bin_path)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.perf_counter() - started, 2)
        raise HermesTimeoutError(
            f'Hermes 分析超时（{int(timeout)}秒）。DeepSeek 可能繁忙，请稍后重试。'
        ) from exc
    except FileNotFoundError as exc:
        raise HermesUnavailableError(
            f'Hermes 命令不可用：无法执行 {bin_path}。'
            '请将 hermes 加入 opc 的 PATH，或设置 KRONOS_HERMES_BIN。'
        ) from exc

    elapsed = round(time.perf_counter() - started, 2)
    # stdout only: stderr may contain provider logs or secrets from the CLI.
    analysis = (completed.stdout or '').strip()
    if completed.returncode != 0:
        raise HermesFailedError(f'Hermes 分析失败（退出码 {completed.returncode}）。')
    if not analysis:
        raise HermesFailedError('Hermes 没有返回分析文本。')
    if len(analysis) > MAX_ANALYSIS_CHARS:
        analysis = analysis[:MAX_ANALYSIS_CHARS].rstrip() + '\n…'
    return {
        'analysis': analysis,
        'model': hermes_model(),
        'provider': hermes_provider(),
        'elapsed_sec': elapsed,
        'cached': False,
    }
