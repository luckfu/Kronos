import os
import pickle
import re
import threading
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
try:
    import torch
except ImportError:
    torch = None
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlencode
warnings.filterwarnings('ignore')

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    bs = None
    BAOSTOCK_AVAILABLE = False

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KRONOS_REMOTE_ONLY = os.getenv('KRONOS_REMOTE_ONLY', '0').lower() in ('1', 'true', 'yes')

try:
    if KRONOS_REMOTE_ONLY:
        raise ImportError('local model disabled by KRONOS_REMOTE_ONLY')
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_SHARE_DATASET_DIR = os.path.join(
    PROJECT_ROOT, 'data', 'a_share_full_market_v1_beta', 'processed_datasets'
)
A_SHARE_RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'a_share', 'a_share_daily.csv')
A_SHARE_SIZE_REFERENCE_PATH = os.path.join(PROJECT_ROOT, 'webui', 'size_reference.json')
A_SHARE_SECTOR_VOCABULARY_PATH = os.getenv(
    'KRONOS_SECTOR_VOCABULARY_PATH',
    os.path.join(PROJECT_ROOT, 'webui', 'sector_vocabulary.json'),
)
A_SHARE_SYMBOL_SECTOR_MAP_PATH = os.getenv(
    'KRONOS_SYMBOL_SECTOR_MAP_PATH',
    os.path.join(PROJECT_ROOT, 'webui', 'symbol_sector_map.json'),
)
MAX_MARKET_DATA_AGE_DAYS = 10
A_SHARE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    'models',
    'a_share_v1_beta',
    'releases',
    'beta_v1.2',
    'best_model',
)
A_SHARE_TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    'models',
    'a_share_v1_beta',
    'releases',
    'beta_v1.2',
    'tokenizer',
)
KRONOS_INFERENCE_URL = os.getenv(
    'KRONOS_INFERENCE_URL',
    'https://luckfu--kronos-beta-v1-2-inference-web.modal.run',
).rstrip('/')
KRONOS_API_KEY = os.getenv('KRONOS_API_KEY')
KRONOS_INFERENCE_TIMEOUT = float(os.getenv('KRONOS_INFERENCE_TIMEOUT', '210'))
KRONOS_MODEL_ID = 'luckfu/Kronos-A-Share-Beta-V1-2'
KRONOS_MODEL_KEY = 'a-share-beta-v1-2'
KRONOS_RELEASE_ID = 'beta-v1.2'
KRONOS_CHECKPOINT = 'Best@871'
KRONOS_NUM_SECTORS = 86
PREDICTION_RESULTS_DIR = os.getenv(
    'KRONOS_HISTORY_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results'),
)
MARKET_DATA_CACHE_DIR = os.getenv(
    'KRONOS_MARKET_DATA_CACHE_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data_cache'),
)
KRONOS_INFERENCE_BACKEND = (
    'remote' if KRONOS_REMOTE_ONLY
    else os.getenv('KRONOS_INFERENCE_BACKEND', 'local').lower()
)
KRONOS_INFERENCE_SEED = int(os.getenv('KRONOS_INFERENCE_SEED', '20260817'))

def selected_backend(value=None):
    """Return the configured inference backend, enforcing server policy."""
    backend = str(value or KRONOS_INFERENCE_BACKEND).lower()
    if KRONOS_REMOTE_ONLY:
        return 'remote'
    return backend


def _seed_inference(seed):
    """Pin the sampling RNG so predictions are reproducible across calls."""
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# Global variables to store models
tokenizer = None
model = None
predictor = None
current_model_key = None
current_model_config = None
a_share_splits = None
model_load_lock = threading.Lock()
baostock_lock = threading.Lock()
inference_lock = threading.Lock()
size_reference_cache = None
size_reference_lock = threading.Lock()
sector_reference_cache = None
sector_reference_lock = threading.Lock()
market_data_cache_lock = threading.Lock()

# Available model configurations
AVAILABLE_MODELS = {
    KRONOS_MODEL_KEY: {
        'name': 'A-share Full-Market Beta V1.2',
        'model_id': A_SHARE_MODEL_PATH,
        'tokenizer_id': A_SHARE_TOKENIZER_PATH,
        'context_length': 512,
        'params': '102.4M',
        'release_id': KRONOS_RELEASE_ID,
        'checkpoint': KRONOS_CHECKPOINT,
        'description': 'Full-market A-share Beta V1.2 with sector and continuous size-percentile conditioning',
        'num_sectors': KRONOS_NUM_SECTORS,
        'num_size_buckets': 0,
        'use_size_percentile': True,
        'default_lookback': 120,
        'default_pred_len': 10,
        'default_temperature': 0.65,
        'default_top_p': 0.8,
        'default_sample_count': 50,
        'model_kwargs': {
            'num_sectors': KRONOS_NUM_SECTORS,
            'num_size_buckets': 0,
            'context_layer': 10,
            'use_size_percentile': True,
            'size_mlp_hidden_dim': 64,
        },
        'local': True,
    }
}


def automatic_device():
    """Choose the accelerator without exposing device selection in the UI."""
    if torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch is not None and torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def ensure_model_loaded():
    """Load the only production model once, on the best available device."""
    global tokenizer, model, predictor, current_model_key, current_model_config
    if predictor is not None:
        return automatic_device()

    with model_load_lock:
        if predictor is not None:
            return automatic_device()
        if not MODEL_AVAILABLE:
            raise RuntimeError('Kronos model library is not available')

        model_key = KRONOS_MODEL_KEY
        model_config = AVAILABLE_MODELS[model_key]
        if not os.path.exists(model_config['model_id']):
            raise FileNotFoundError(f'Local checkpoint not found: {model_config["model_id"]}')

        device = automatic_device()
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id']).eval()
        model = Kronos.from_pretrained(
            model_config['model_id'],
            **model_config.get('model_kwargs', {}),
        ).eval()
        predictor = KronosPredictor(
            model,
            tokenizer,
            device=device,
            max_context=model_config['context_length'],
        )
        current_model_key = model_key
        current_model_config = model_config
        return device


def read_baostock_result(result):
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock query failed: {result.error_code} {result.error_msg}')
    return rows


stock_name_cache = {}

MARKET_DATA_COLUMNS = [
    'timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'turn', 'pctChg',
]


def _market_cache_path(symbol):
    """Return a safe per-symbol path for the incremental market cache."""
    normalized = normalize_a_share_symbol(symbol)
    if not re.fullmatch(r'(sh|sz|bj)\.\d{6}', normalized):
        raise ValueError(f'Invalid A-share symbol for market cache: {symbol}')
    return os.path.join(MARKET_DATA_CACHE_DIR, f'{normalized.replace(".", "_")}.csv')


def _read_market_data_cache(symbol):
    path = _market_cache_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame(columns=MARKET_DATA_COLUMNS)
    try:
        frame = pd.read_csv(path, parse_dates=['timestamps'])
        if not set(MARKET_DATA_COLUMNS).issubset(frame.columns):
            return pd.DataFrame(columns=MARKET_DATA_COLUMNS)
        frame = frame[MARKET_DATA_COLUMNS].copy()
        for column in MARKET_DATA_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        frame = frame.dropna(subset=MARKET_DATA_COLUMNS)
        return frame.sort_values('timestamps').drop_duplicates(
            'timestamps', keep='last'
        ).reset_index(drop=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame(columns=MARKET_DATA_COLUMNS)


def _merge_market_data_cache(symbol, frame):
    """Merge fresh rows into the per-symbol cache atomically."""
    with market_data_cache_lock:
        if frame is None or frame.empty:
            return _read_market_data_cache(symbol)
        merged = pd.concat([_read_market_data_cache(symbol), frame], ignore_index=True)
        merged['timestamps'] = pd.to_datetime(merged['timestamps'], errors='coerce')
        for column in MARKET_DATA_COLUMNS[1:]:
            merged[column] = pd.to_numeric(merged[column], errors='coerce')
        merged = merged.dropna(subset=MARKET_DATA_COLUMNS)
        merged = merged.sort_values('timestamps').drop_duplicates(
            'timestamps', keep='last'
        ).reset_index(drop=True)
        path = _market_cache_path(symbol)
        os.makedirs(MARKET_DATA_CACHE_DIR, exist_ok=True)
        temporary_path = f'{path}.tmp.{os.getpid()}'
        merged.to_csv(temporary_path, index=False, date_format='%Y-%m-%d')
        os.replace(temporary_path, path)
        return merged


def query_baostock_stock_name(symbol):
    """Resolve a stock name through the same market-data service as quotes."""
    if not BAOSTOCK_AVAILABLE:
        return None
    try:
        with baostock_lock:
            login = bs.login()
            if getattr(login, 'error_code', '1') != '0':
                return None
            try:
                result = bs.query_stock_basic(code=str(symbol).strip().lower())
                if getattr(result, 'error_code', '1') != '0' or not result.next():
                    return None
                row = result.get_row_data()
                return str(row[1]).strip() if len(row) > 1 and row[1] else None
            finally:
                bs.logout()
    except Exception:
        return None


def query_remote_stock_name(symbol):
    """Resolve a Chinese display name with BaoStock and a network fallback."""
    normalized_symbol = str(symbol).strip().lower()
    if normalized_symbol in stock_name_cache:
        return stock_name_cache[normalized_symbol]
    if '.' not in normalized_symbol:
        return None
    name = query_baostock_stock_name(normalized_symbol)
    if name:
        stock_name_cache[normalized_symbol] = name
        return name
    market, code = normalized_symbol.split('.', 1)
    secid = f"1.{code}" if market == 'sh' else f"0.{code}"
    query = urlencode({'secid': secid, 'fields': 'f57,f58'})
    url = f'https://push2.eastmoney.com/api/qt/stock/get?{query}'
    try:
        response = subprocess.run(
            ['curl', '--fail', '--silent', '--show-error', '--compressed',
             '--max-time', '10', '-A', 'Mozilla/5.0', url],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        payload = json.loads(response.stdout)
        name = ((payload.get('data') or {}).get('f58') or '').strip()
        if name:
            stock_name_cache[normalized_symbol] = name
        return name or None
    except Exception:
        return None


def require_fresh_market_data(history, source):
    """Reject suspended, delisted, or stale histories in production prediction."""
    if history.empty:
        raise RuntimeError(f'{source} returned no usable daily data')
    latest_date = pd.Timestamp(history['timestamps'].iloc[-1]).normalize()
    age_days = (pd.Timestamp.now().normalize() - latest_date).days
    if age_days > MAX_MARKET_DATA_AGE_DAYS:
        raise RuntimeError(
            f'{source} latest trading date is {latest_date:%Y-%m-%d}; '
            'the stock may be delisted or long-term suspended'
        )


def query_latest_daily_data(symbol, lookback, pred_len, history_rows=None):
    """Incrementally refresh the cached adjusted context via BaoStock."""
    if not BAOSTOCK_AVAILABLE:
        raise RuntimeError('BaoStock is not installed')

    retained_rows = max(lookback, int(history_rows or lookback))
    today = pd.Timestamp.now().normalize()
    cached = _read_market_data_cache(symbol)
    history_start = (
        pd.Timestamp(cached['timestamps'].iloc[-1]) - pd.Timedelta(days=7)
        if not cached.empty else today - pd.Timedelta(days=max(365, retained_rows * 4))
    )
    fields = 'date,code,open,high,low,close,volume,amount,turn,pctChg'

    with baostock_lock:
        login = bs.login()
        if login.error_code != '0':
            raise RuntimeError(f'BaoStock login failed: {login.error_code} {login.error_msg}')
        try:
            history_result = bs.query_history_k_data_plus(
                symbol,
                fields,
                start_date=history_start.strftime('%Y-%m-%d'),
                end_date=today.strftime('%Y-%m-%d'),
                frequency='d',
                adjustflag='2',
            )
            history_rows = read_baostock_result(history_result)
            history = pd.DataFrame(history_rows, columns=fields.split(','))
            history = history.rename(columns={'date': 'timestamps'})
            history['timestamps'] = pd.to_datetime(history['timestamps'], errors='coerce')
            numeric_columns = [
                'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg'
            ]
            for column in numeric_columns:
                history[column] = pd.to_numeric(history[column], errors='coerce')
            history = history.dropna(subset=['timestamps', *numeric_columns])
            history = history[(history['close'] > 0) & (history['volume'] >= 0)]
            history = history.sort_values('timestamps').drop_duplicates('timestamps', keep='last')
            history = _merge_market_data_cache(symbol, history)
            history = history.tail(retained_rows).reset_index(drop=True)
            require_fresh_market_data(history, 'BaoStock')

            last_date = history['timestamps'].iloc[-1]
            calendar_end = last_date + pd.Timedelta(days=max(45, pred_len * 6))
            calendar_result = bs.query_trade_dates(
                start_date=(last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                end_date=calendar_end.strftime('%Y-%m-%d'),
            )
            calendar_rows = read_baostock_result(calendar_result)
            future_dates = [
                pd.Timestamp(row[0]) for row in calendar_rows
                if len(row) > 1 and row[1] == '1'
            ][:pred_len]
            if len(future_dates) < pred_len:
                raise RuntimeError(
                    f'BaoStock returned only {len(future_dates)} future trading dates'
                )
            return history, pd.Series(future_dates, name='timestamps')
        finally:
            bs.logout()


def query_eastmoney_daily_data(symbol, lookback, pred_len, history_rows=None):
    """Incrementally refresh the cached adjusted daily context from Eastmoney."""
    retained_rows = max(lookback, int(history_rows or lookback))
    today = pd.Timestamp.now().normalize()
    cached = _read_market_data_cache(symbol)
    history_start = (
        pd.Timestamp(cached['timestamps'].iloc[-1]) - pd.Timedelta(days=7)
        if not cached.empty else today - pd.Timedelta(days=max(365, retained_rows * 4))
    )
    market, code = str(symbol).lower().split('.', 1)
    secid = f"1.{code}" if market == 'sh' else f"0.{code}"
    query = urlencode({
        'secid': secid,
        'klt': '101',
        'fqt': '1',
        'beg': history_start.strftime('%Y%m%d'),
        'end': today.strftime('%Y%m%d'),
        'fields1': 'f1',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
    })
    url = f'https://push2his.eastmoney.com/api/qt/stock/kline/get?{query}'
    response = subprocess.run(
        [
            'curl', '--fail', '--silent', '--show-error', '--compressed',
            '--max-time', '20', '-A', 'Mozilla/5.0', url,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=25,
    )
    payload = json.loads(response.stdout)
    data = payload.get('data') or {}
    klines = data.get('klines') or []
    columns = [
        'timestamps', 'open', 'close', 'high', 'low', 'volume', 'amount',
        'amplitude', 'pctChg', 'change', 'turn',
    ]
    history = pd.DataFrame(
        [row.split(',') for row in klines], columns=columns
    )
    history['timestamps'] = pd.to_datetime(history['timestamps'], errors='coerce')
    numeric_columns = [
        'open', 'close', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg',
    ]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors='coerce')
    history = history.dropna(subset=['timestamps', *numeric_columns])
    history = history[(history['close'] > 0) & (history['volume'] >= 0)]
    history = _merge_market_data_cache(symbol, history)
    history = history.tail(retained_rows).reset_index(drop=True)
    history = _merge_market_data_cache(symbol, history)
    history = history.tail(retained_rows).reset_index(drop=True)
    if len(history) < lookback:
        raise RuntimeError(
            f'Eastmoney returned only {len(history)} usable rows; {lookback} are required'
        )
    require_fresh_market_data(history, 'Eastmoney')
    last_date = pd.Timestamp(history['timestamps'].iloc[-1])
    future_dates = pd.Series(
        pd.bdate_range(
            last_date + pd.Timedelta(days=1),
            periods=pred_len,
        ),
        name='timestamps',
    )
    return history, future_dates


def load_latest_size_reference():
    """Load the full-market cross-section used for continuous size ranks."""
    global size_reference_cache
    if size_reference_cache is not None:
        return size_reference_cache

    with size_reference_lock:
        if size_reference_cache is not None:
            return size_reference_cache
        if os.path.exists(A_SHARE_SIZE_REFERENCE_PATH):
            with open(A_SHARE_SIZE_REFERENCE_PATH, encoding='utf-8') as handle:
                payload = json.load(handle)
            reference_date = pd.Timestamp(payload['reference_date'])
            market_caps = np.sort(np.asarray(payload['market_caps'], dtype=np.float64))
        elif os.path.exists(A_SHARE_RAW_DATA_PATH):
            frame = pd.read_csv(
                A_SHARE_RAW_DATA_PATH,
                usecols=['date', 'market_cap'],
                parse_dates=['date'],
            )
            frame['market_cap'] = pd.to_numeric(frame['market_cap'], errors='coerce')
            frame = frame.dropna(subset=['date', 'market_cap'])
            frame = frame[frame['market_cap'] > 0]
            if frame.empty:
                raise ValueError('A-share market-cap reference panel is empty')
            reference_date = pd.Timestamp(frame['date'].max())
            market_caps = np.sort(
                frame.loc[frame['date'] == reference_date, 'market_cap'].to_numpy(dtype=np.float64)
            )
        else:
            raise FileNotFoundError(
                f'Market-cap reference not found: {A_SHARE_SIZE_REFERENCE_PATH}'
            )
        if len(market_caps) < 100:
            raise ValueError(
                f'Only {len(market_caps)} market-cap references are available on {reference_date:%Y-%m-%d}'
            )
        size_reference_cache = {
            'date': reference_date,
            'market_caps': market_caps,
            'count': int(len(market_caps)),
        }
        return size_reference_cache


def load_sector_reference():
    """Load the immutable model vocabulary and replaceable symbol mapping."""
    global sector_reference_cache
    if sector_reference_cache is not None:
        return sector_reference_cache

    with sector_reference_lock:
        if sector_reference_cache is not None:
            return sector_reference_cache
        with open(A_SHARE_SECTOR_VOCABULARY_PATH, encoding='utf-8') as handle:
            vocabulary = json.load(handle)
        with open(A_SHARE_SYMBOL_SECTOR_MAP_PATH, encoding='utf-8') as handle:
            mapping = json.load(handle)
        labels = vocabulary.get('sector_labels') or []
        if len(labels) != KRONOS_NUM_SECTORS:
            raise ValueError(
                f'Sector reference has {len(labels)} labels; {KRONOS_NUM_SECTORS} are required'
            )
        if len(set(labels)) != len(labels):
            raise ValueError('Sector vocabulary contains duplicate labels')
        unknown_sector_id = int(
            vocabulary.get('unknown_sector_id', KRONOS_NUM_SECTORS)
        )
        if unknown_sector_id != KRONOS_NUM_SECTORS:
            raise ValueError('Sector reference unknown ID does not match the model contract')
        vocabulary_id = str(vocabulary.get('vocabulary_id') or '')
        if not vocabulary_id or mapping.get('vocabulary_id') != vocabulary_id:
            raise ValueError('Symbol-sector mapping does not match the model vocabulary')
        label_to_id = {str(value): index for index, value in enumerate(labels)}
        symbols = {}
        for symbol, raw_record in (mapping.get('symbols') or {}).items():
            sector_label = str(raw_record.get('sector_label') or 'unknown')
            sector_id = int(raw_record['sector_id'])
            expected_id = label_to_id.get(sector_label, unknown_sector_id)
            if sector_id != expected_id:
                raise ValueError(
                    f'Symbol-sector mapping has an invalid ID for {symbol}: '
                    f'{sector_id} != {expected_id}'
                )
            reference_date = pd.Timestamp(raw_record['reference_date'])
            if pd.isna(reference_date):
                raise ValueError(f'Symbol-sector mapping has an invalid date for {symbol}')
            symbols[str(symbol)] = {
                'sector_id': sector_id,
                'sector_label': sector_label,
                'reference_date': reference_date.date().isoformat(),
            }
        sector_reference_cache = {
            'date': pd.Timestamp(mapping['reference_date']),
            'vocabulary_id': vocabulary_id,
            'labels': tuple(str(value) for value in labels),
            'label_to_id': label_to_id,
            'unknown_sector_id': unknown_sector_id,
            'symbols': symbols,
        }
        return sector_reference_cache


def sector_condition_from_local_frame(local_frame, asof=None):
    """Resolve a point-in-time sector from a prepared local symbol frame."""
    if local_frame is None or 'sector' not in local_frame.columns:
        return None
    rows = local_frame.dropna(subset=['sector']).copy()
    if asof is not None:
        rows = rows[rows['timestamps'] <= pd.Timestamp(asof)]
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    label = str(latest['sector'])
    reference = load_sector_reference()
    sector_id = int(reference['label_to_id'].get(label, reference['unknown_sector_id']))
    return {
        'sector_id': sector_id,
        'sector_label': label,
        'sector_reference_date': pd.Timestamp(latest['timestamps']),
        'sector_source': 'local_point_in_time_panel',
    }


def sector_condition_for_symbol(symbol, asof, local_frame=None):
    """Resolve the latest non-future sector, falling back to the unknown ID."""
    local = sector_condition_from_local_frame(local_frame, asof=asof)
    if local is not None:
        return local

    reference = load_sector_reference()
    record = reference['symbols'].get(normalize_a_share_symbol(symbol))
    if record is not None:
        reference_date = pd.Timestamp(record['reference_date'])
        if reference_date <= pd.Timestamp(asof):
            sector_id = int(record['sector_id'])
            if not 0 <= sector_id < KRONOS_NUM_SECTORS:
                sector_id = reference['unknown_sector_id']
            return {
                'sector_id': sector_id,
                'sector_label': str(record['sector_label']),
                'sector_reference_date': reference_date,
                'sector_source': 'portable_sector_snapshot',
            }
    return {
        'sector_id': reference['unknown_sector_id'],
        'sector_label': 'unknown',
        'sector_reference_date': None,
        'sector_source': 'unknown_sector_fallback',
    }


def size_condition_from_remote_context(context):
    """Map amount/turnover market-cap proxy to the training cross-section."""
    ordered = context.sort_values('timestamps').copy()
    for column in ('amount', 'turn'):
        ordered[column] = pd.to_numeric(ordered[column], errors='coerce')
    usable = ordered[
        np.isfinite(ordered['amount'])
        & np.isfinite(ordered['turn'])
        & (ordered['amount'] > 0)
        & (ordered['turn'] > 0)
    ]
    if usable.empty:
        raise ValueError('Market history has no usable amount/turnover row for size conditioning')
    latest = usable.iloc[-1]
    amount = float(latest['amount'])
    turnover = float(latest['turn'])
    market_cap = amount / (turnover / 100.0)
    if not np.isfinite(market_cap) or market_cap <= 0:
        raise ValueError('Unable to estimate a positive float market cap from BaoStock')

    reference = load_latest_size_reference()
    percentile = float(
        np.searchsorted(reference['market_caps'], market_cap, side='right')
        / reference['count']
    )
    percentile = float(np.clip(percentile, 0.0, 1.0))
    return {
        'size_percentile': percentile,
        'size_percentile_asof': pd.Timestamp(latest['timestamps']),
        'size_reference_date': reference['date'],
        'size_source': 'amount_turnover_proxy_vs_full_market_cross_section',
        'estimated_market_cap': float(market_cap),
    }


def size_condition_from_local_frame(local_frame, asof=None):
    if local_frame is None or 'size_percentile' not in local_frame.columns:
        return None
    rows = local_frame.dropna(subset=['size_percentile']).copy()
    if asof is not None:
        rows = rows[rows['timestamps'] <= pd.Timestamp(asof)]
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    percentile = float(latest['size_percentile'])
    asof = pd.Timestamp(latest['timestamps'])
    return {
        'size_percentile': float(np.clip(percentile, 0.0, 1.0)),
        'size_percentile_asof': asof,
        'size_reference_date': asof,
        'size_source': 'local_point_in_time_panel',
        'estimated_market_cap': None,
    }


def latest_prediction_inputs(symbol, lookback, pred_len, history_rows=None):
    """Refresh a stock and derive the two Beta V1.2 conditions."""
    symbol = normalize_a_share_symbol(symbol)
    retained_rows = max(lookback, int(history_rows or lookback))
    local_frame, local_error = get_a_share_symbol_frame(symbol)
    local_size = size_condition_from_local_frame(local_frame)

    refresh_errors = []
    try:
        context, future_dates = query_eastmoney_daily_data(
            symbol,
            lookback,
            pred_len,
            history_rows=retained_rows,
        )
        try:
            size_context = size_condition_from_remote_context(context)
        except Exception:
            if local_size is None:
                raise
            size_context = local_size
        data_source = 'eastmoney'
        calendar_source = 'business_day_fallback'
        refresh_error = None
    except Exception as exc:
        refresh_errors.append(f'Eastmoney: {exc}')
        try:
            context, future_dates = query_latest_daily_data(
                symbol,
                lookback,
                pred_len,
                history_rows=retained_rows,
            )
            try:
                size_context = size_condition_from_remote_context(context)
            except Exception:
                if local_size is None:
                    raise
                size_context = local_size
            data_source = 'baostock'
            calendar_source = 'baostock'
            refresh_error = '; '.join(refresh_errors)
        except Exception as baostock_exc:
            refresh_errors.append(f'BaoStock: {baostock_exc}')
            local_date = None
            if local_frame is not None and not local_frame.empty:
                local_date = pd.Timestamp(local_frame['timestamps'].iloc[-1])
            cache_note = (
                f'；本地缓存仅到 {local_date:%Y-%m-%d}，生产预测已禁止使用旧缓存'
                if local_date is not None else ''
            )
            raise ValueError(
                f'{symbol} 无法获取最新行情，可能已退市、长期停牌或数据源暂不可用'
                f'{cache_note}。详情：{"; ".join(refresh_errors)}'
            ) from baostock_exc

    if len(context) < lookback:
        raise ValueError(f'{symbol} has only {len(context)} rows; {lookback} are required')
    signal_date = pd.Timestamp(context['timestamps'].iloc[-1])
    sector_context = sector_condition_for_symbol(
        symbol,
        asof=signal_date,
        local_frame=local_frame,
    )
    return {
        'context': context.tail(retained_rows).reset_index(drop=True),
        'future_dates': future_dates,
        'stock_name': query_remote_stock_name(symbol),
        **size_context,
        **sector_context,
        'in_local_panel': local_frame is not None,
        'local_panel_error': local_error,
        'data_source': data_source,
        'calendar_source': calendar_source,
        'refresh_error': refresh_error,
    }


def load_a_share_splits():
    """Load the prepared A-share panel once and keep it indexed by symbol."""
    global a_share_splits
    if a_share_splits is not None:
        return a_share_splits

    splits = {}
    for split in ('train', 'val'):
        path = os.path.join(A_SHARE_DATASET_DIR, f'{split}_data.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f'A-share dataset not found: {path}')
        with open(path, 'rb') as handle:
            splits[split] = pickle.load(handle)
    a_share_splits = splits
    return a_share_splits


def normalize_a_share_symbol(symbol):
    """Accept both Qlib symbols (sz.300395) and six-digit A-share codes."""
    value = str(symbol or '').strip().lower()
    if '.' in value or not value.isdigit() or len(value) != 6:
        return value
    if value.startswith(('600', '601', '603', '605', '688', '689')):
        return f'sh.{value}'
    if value.startswith(('000', '001', '002', '003', '300', '301')):
        return f'sz.{value}'
    if value.startswith(('400', '430', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '870', '871', '872', '873')):
        return f'bj.{value}'
    return value


def get_a_share_symbol_frame(symbol):
    """Return one symbol's train and validation history as a normal DataFrame."""
    symbol = normalize_a_share_symbol(symbol)
    try:
        splits = load_a_share_splits()
    except FileNotFoundError as exc:
        return None, str(exc)
    frames = [
        split[symbol]
        for split in (splits['train'], splits['val'])
        if symbol in split
    ]
    if not frames:
        return None, f'Unknown A-share symbol: {symbol}'

    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep='last')].reset_index()
    date_column = 'datetime' if 'datetime' in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_column: 'timestamps'})
    frame['timestamps'] = pd.to_datetime(frame['timestamps'])
    return frame, None


def forecast_return_summary(close_samples, latest_close):
    """Summarize the forecast and classify its 10-day timing direction."""
    close_samples = np.asarray(close_samples, dtype=np.float64)
    if close_samples.ndim != 2 or close_samples.shape[1] == 0:
        raise ValueError('Forecast close samples must have shape [samples, horizon]')
    if not np.isfinite(latest_close) or latest_close <= 0:
        raise ValueError('Latest close must be positive')
    average_closes = close_samples.mean(axis=1)
    returns = average_closes / latest_close - 1
    predicted_return_p50 = float(np.median(returns))
    if predicted_return_p50 >= 0.01:
        timing_signal = {'key': 'bullish', 'label': '偏多'}
    elif predicted_return_p50 <= -0.01:
        timing_signal = {'key': 'bearish', 'label': '偏空'}
    else:
        timing_signal = {'key': 'neutral', 'label': '观望'}
    return {
        'predicted_average_close_p50': float(np.median(average_closes)),
        'predicted_return_p50': predicted_return_p50,
        'positive_path_rate': float(np.mean(returns > 0)),
        'timing_signal': timing_signal,
    }


def call_remote_inference(payload, endpoint='predict', expected_key='predictions'):
    """Call the model-only Modal service without moving market collection there."""
    body = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if KRONOS_API_KEY:
        headers['Authorization'] = f'Bearer {KRONOS_API_KEY}'
    request_obj = urllib.request.Request(
        f'{KRONOS_INFERENCE_URL}/{endpoint}', data=body, headers=headers, method='POST'
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=KRONOS_INFERENCE_TIMEOUT) as response:
            response_body = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Modal inference HTTP {exc.code}: {detail}') from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f'Modal inference request failed: {exc}') from exc
    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError('Modal inference returned invalid JSON') from exc
    if not isinstance(result, dict) or expected_key not in result:
        raise RuntimeError(f'Modal inference returned an invalid response: {result}')
    return result


def portfolio_ranking_batch(symbols, lookback, pred_len):
    """Collect and align several stocks locally for comparable batch inference."""
    collected = [
        (symbol, latest_prediction_inputs(symbol, lookback, pred_len, history_rows=lookback + 20))
        for symbol in symbols
    ]
    as_of_date = min(pd.Timestamp(item['context']['timestamps'].iloc[-1]) for _, item in collected)
    records = []
    x_rows = []
    y_stamps = []
    for symbol, item in collected:
        context = item['context'].loc[item['context']['timestamps'] <= as_of_date].tail(lookback).reset_index(drop=True)
        if len(context) != lookback:
            raise ValueError(f'{symbol} 在共同截止日之前不足 {lookback} 个交易日')
        future_dates = pd.Series(pd.bdate_range(as_of_date + pd.Timedelta(days=1), periods=pred_len), name='timestamps')
        size_context = size_condition_from_remote_context(context)
        sector_context = sector_condition_for_symbol(symbol, asof=as_of_date)
        record = {
            **item,
            **size_context,
            **sector_context,
            'symbol': symbol,
            'context': context,
            'future_dates': future_dates,
        }
        records.append(record)
        x_rows.append(context[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(dtype=np.float32))
        y_stamps.append(pd.DataFrame({
            'minute': future_dates.dt.minute, 'hour': future_dates.dt.hour,
            'weekday': future_dates.dt.weekday, 'day': future_dates.dt.day, 'month': future_dates.dt.month,
        }).to_numpy(dtype=np.float32))
    return {'as_of_date': as_of_date, 'x': np.stack(x_rows), 'y_stamp': np.stack(y_stamps), 'records': records}


def local_inference(
    context, future_dates, pred_len, temperature, top_p, sample_count,
    sector_id, size_percentile,
):
    """Run Beta V1.2 locally with both required conditioning inputs."""
    ensure_model_loaded()
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    x_df = context[feature_columns]
    x_timestamp = pd.Series(pd.to_datetime(context['timestamps']).to_numpy(), name='timestamps')
    y_timestamp = pd.Series(pd.to_datetime(future_dates).to_numpy(), name='timestamps')
    with inference_lock:
        _seed_inference(KRONOS_INFERENCE_SEED)
        prediction_samples = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=top_p, sample_count=sample_count,
            verbose=False, sector_id=sector_id, size_bucket=None,
            size_percentile=size_percentile, return_samples=True,
        )
    expected_shape = (sample_count, pred_len, len(feature_columns))
    if prediction_samples.shape != expected_shape:
        raise RuntimeError(f'Unexpected prediction sample shape: {prediction_samples.shape}')
    mean_prediction = prediction_samples.mean(axis=0)
    pred_df = pd.DataFrame(mean_prediction, columns=feature_columns, index=pd.DatetimeIndex(y_timestamp))
    pred_df['high'] = pred_df[['open', 'high', 'low', 'close']].max(axis=1)
    pred_df['low'] = pred_df[['open', 'high', 'low', 'close']].min(axis=1)
    pred_df['volume'] = pred_df['volume'].clip(lower=0)
    pred_df['amount'] = pred_df['amount'].clip(lower=0)
    close_samples = prediction_samples[:, :, feature_columns.index('close')]
    close_quantiles = np.quantile(close_samples, [0.1, 0.5, 0.9], axis=0)
    prediction_results = []
    for index, timestamp in enumerate(y_timestamp):
        row = pred_df.iloc[index]
        prediction_results.append({
            'timestamp': pd.Timestamp(timestamp).isoformat(),
            **{column: float(row[column]) for column in feature_columns},
            'close_p10': float(close_quantiles[0, index]),
            'close_p50': float(close_quantiles[1, index]),
            'close_p90': float(close_quantiles[2, index]),
        })
    return prediction_results, pred_df, pd.DataFrame({
        'close_p10': close_quantiles[0], 'close_p50': close_quantiles[1], 'close_p90': close_quantiles[2]
    }, index=pd.DatetimeIndex(y_timestamp)), close_samples, str(next(predictor.model.parameters()).device)


def resolve_request_data(data):
    """Resolve either a built-in A-share symbol or a user-selected data file."""
    symbol = normalize_a_share_symbol(data.get('symbol'))
    if symbol:
        frame, error = get_a_share_symbol_frame(symbol)
        return frame, error, f'a-share:{symbol}', symbol

    file_path = data.get('file_path')
    if not file_path:
        return None, 'Select an A-share symbol or a data file', None, None
    frame, error = load_data_file(file_path)
    return frame, error, file_path, None

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = PREDICTION_RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'prediction_{timestamp}_{uuid.uuid4().hex[:8]}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None


def save_prediction_record(record):
    """Persist the complete response needed to reopen a prediction later."""
    results_dir = Path(PREDICTION_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    payload = {**record, 'record_id': record_id, 'created_at': created_at}
    path = results_dir / f'{record_id}.json'
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'), default=str)
    return payload


def prediction_signal_date(payload):
    """Return the market-data cutoff used to group a cross-section."""
    params = payload.get('prediction_params') or {}
    value = (
        payload.get('latest_data_date')
        or params.get('latest_data_date')
        or payload.get('created_at')
        or payload.get('timestamp')
    )
    try:
        parsed = pd.Timestamp(value)
        return None if pd.isna(parsed) else parsed.date().isoformat()
    except (TypeError, ValueError):
        return None


def prediction_record_summary(payload):
    predictions = payload.get('prediction_results') or []
    params = payload.get('prediction_params') or {}
    symbol = payload.get('symbol') or params.get('symbol')
    name = payload.get('name')
    # Older records could persist the normalized symbol as the name when the
    # quote endpoint was unavailable. Re-query those records so the history
    # view can recover the real Chinese name without rewriting old files.
    if name and symbol and str(name).strip().lower() == str(symbol).strip().lower():
        name = None
    if not name and symbol:
        name = query_remote_stock_name(symbol)
    return {
        'record_id': payload.get('record_id'),
        'created_at': payload.get('created_at') or payload.get('timestamp'),
        'signal_date': prediction_signal_date(payload),
        'latest_data_date': payload.get('latest_data_date'),
        'symbol': symbol,
        'name': name,
        'prediction_type': payload.get('prediction_type'),
        'latest_close': payload.get('latest_close'),
        'predicted_return_p50': payload.get('predicted_return_p50'),
        'positive_path_rate': payload.get('positive_path_rate'),
        'timing_signal': payload.get('timing_signal') or {'key': 'neutral', 'label': '观望'},
        'final_close_p50': payload.get('final_close_p50'),
        'pred_len': payload.get('pred_len') or len(predictions),
        'backend': payload.get('backend', 'remote'),
        'sector_id': payload.get('sector_id'),
        'sector_label': payload.get('sector_label'),
        'size_percentile': payload.get('size_percentile'),
        'interval': payload.get('interval'),
    }


@app.route('/api/prediction-history')
def prediction_history():
    """List saved prediction summaries; Nginx protects this endpoint in production."""
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    symbol_filter = request.args.get('symbol', '').strip().lower()
    direction = request.args.get('direction', '').strip().lower()
    backend = request.args.get('backend', '').strip().lower()
    for value, label in ((date_from, '开始日期'), (date_to, '结束日期')):
        if value:
            try:
                datetime.date.fromisoformat(value)
            except ValueError:
                return jsonify({'error': f'{label}格式无效，应为 YYYY-MM-DD'}), 400
    if date_from and date_to and date_from > date_to:
        return jsonify({'error': '开始日期不能晚于结束日期'}), 400
    if direction not in ('', 'positive', 'negative'):
        return jsonify({'error': '无效的涨跌方向筛选'}), 400
    if backend not in ('', 'remote', 'local'):
        return jsonify({'error': '无效的推理来源筛选'}), 400
    results_dir = Path(PREDICTION_RESULTS_DIR)
    if not results_dir.exists():
        return jsonify({'records': [], 'total': 0})
    records = []
    for path in sorted(results_dir.glob('*.json'), reverse=True):
        try:
            with path.open(encoding='utf-8') as handle:
                payload = json.load(handle)
                if payload.get('record_id') and payload.get('chart'):
                    summary = prediction_record_summary(payload)
                    if summary.get('name') and payload.get('name') != summary['name']:
                        payload['name'] = summary['name']
                        try:
                            with path.open('w', encoding='utf-8') as handle:
                                json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'), default=str)
                        except OSError:
                            pass
                    signal_date = summary.get('signal_date') or ''
                    record_symbol = str(summary.get('symbol') or '').lower()
                    record_return = summary.get('predicted_return_p50')
                    if date_from and signal_date < date_from:
                        continue
                    if date_to and signal_date > date_to:
                        continue
                    if symbol_filter and symbol_filter not in record_symbol:
                        continue
                    if direction == 'positive' and (record_return is None or record_return < 0):
                        continue
                    if direction == 'negative' and (record_return is None or record_return >= 0):
                        continue
                    if backend and summary.get('backend', 'remote') != backend:
                        continue
                    records.append(summary)
        except (OSError, ValueError, TypeError):
            continue
    latest_by_cross_section_symbol = {}
    for summary in records:
        key = (summary.get('signal_date') or '', str(summary.get('symbol') or '').lower())
        previous = latest_by_cross_section_symbol.get(key)
        if previous is None or str(summary.get('created_at') or '') > str(previous.get('created_at') or ''):
            latest_by_cross_section_symbol[key] = summary
    records = sorted(
        latest_by_cross_section_symbol.values(),
        key=lambda summary: str(summary.get('created_at') or ''),
        reverse=True,
    )
    return jsonify({'records': records[:100], 'total': len(records)})


@app.route('/api/prediction-history/<record_id>')
def prediction_history_detail(record_id):
    """Return one saved prediction, including its chart and forecast rows."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', record_id):
        return jsonify({'error': '无效的记录编号'}), 400
    path = Path(PREDICTION_RESULTS_DIR) / f'{record_id}.json'
    if not path.is_file():
        return jsonify({'error': '预测记录不存在'}), 404
    try:
        with path.open(encoding='utf-8') as handle:
            return jsonify(json.load(handle))
    except (OSError, ValueError) as exc:
        return jsonify({'error': f'读取预测记录失败: {exc}'}), 500


@app.route('/api/prediction-history/<record_id>', methods=['DELETE'])
def delete_prediction_history(record_id):
    """Delete one saved prediction record by its generated identifier."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', record_id):
        return jsonify({'error': '无效的记录编号'}), 400
    path = Path(PREDICTION_RESULTS_DIR) / f'{record_id}.json'
    if not path.is_file():
        return jsonify({'error': '预测记录不存在'}), 404
    try:
        path.unlink()
    except OSError as exc:
        return jsonify({'error': f'删除预测记录失败: {exc}'}), 500
    return jsonify({'deleted': True, 'record_id': record_id})


@app.route('/api/prediction-history/dates/<signal_date>', methods=['DELETE'])
def delete_prediction_history_date(signal_date):
    """Delete every saved prediction belonging to one signal-date cross-section."""
    try:
        signal_date = datetime.date.fromisoformat(signal_date).isoformat()
    except ValueError:
        return jsonify({'error': '无效的信号日期'}), 400
    results_dir = Path(PREDICTION_RESULTS_DIR)
    if not results_dir.exists():
        return jsonify({'error': '预测截面不存在'}), 404

    paths = []
    for path in results_dir.glob('*.json'):
        try:
            with path.open(encoding='utf-8') as handle:
                if prediction_signal_date(json.load(handle)) == signal_date:
                    paths.append(path)
        except (OSError, ValueError, TypeError):
            continue
    if not paths:
        return jsonify({'error': '预测截面不存在'}), 404

    try:
        for path in paths:
            path.unlink()
    except OSError as exc:
        return jsonify({'error': f'删除预测截面失败: {exc}'}), 500
    return jsonify({
        'deleted': True,
        'signal_date': signal_date,
        'records_deleted': len(paths),
    })


@app.route('/api/prediction-history/dates/<signal_date>/symbols/<symbol>', methods=['DELETE'])
def delete_prediction_history_symbol(signal_date, symbol):
    """Delete one stock, including duplicates, from a signal-date cross-section."""
    try:
        signal_date = datetime.date.fromisoformat(signal_date).isoformat()
    except ValueError:
        return jsonify({'error': '无效的信号日期'}), 400
    symbol = normalize_a_share_symbol(symbol)
    if not re.fullmatch(r'(?:sh|sz|bj)\.\d{6}', symbol):
        return jsonify({'error': '无效的股票代码'}), 400

    results_dir = Path(PREDICTION_RESULTS_DIR)
    if not results_dir.exists():
        return jsonify({'error': '截面中不存在该股票'}), 404
    paths = []
    for path in results_dir.glob('*.json'):
        try:
            with path.open(encoding='utf-8') as handle:
                payload = json.load(handle)
            if (
                prediction_signal_date(payload) == signal_date
                and normalize_a_share_symbol(payload.get('symbol')) == symbol
            ):
                paths.append(path)
        except (OSError, ValueError, TypeError):
            continue
    if not paths:
        return jsonify({'error': '截面中不存在该股票'}), 404

    try:
        for path in paths:
            path.unlink()
    except OSError as exc:
        return jsonify({'error': f'删除个股预测失败: {exc}'}), 500
    return jsonify({
        'deleted': True,
        'signal_date': signal_date,
        'symbol': symbol,
        'records_deleted': len(paths),
    })

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name=f'历史行情（{len(historical_df)}日）',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if isinstance(pred_df.index, pd.DatetimeIndex):
            pred_timestamps = pred_df.index
        elif 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name=f'预测行情（{len(pred_df)}日）',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'timestamps' in actual_df.columns:
                actual_timestamps = pd.DatetimeIndex(actual_df['timestamps'])
            elif 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name=f'真实行情（{len(actual_df)}日）',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Kronos 预测结果：{len(historical_df)} 日历史 + {len(pred_df) if pred_df is not None else 0} 日预测',
        xaxis_title='交易日',
        yaxis_title='价格',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return pio.to_json(fig, pretty=False)


def create_operational_chart(context, pred_df, interval_df):
    """Create an A-share candlestick chart with uncertainty and volume."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.74, 0.26],
    )
    history_dates = pd.DatetimeIndex(context['timestamps'])
    forecast_dates = pd.DatetimeIndex(pred_df.index)
    plotted_dates = history_dates.union(forecast_dates).normalize()
    calendar_weekdays = pd.date_range(
        plotted_dates.min(),
        plotted_dates.max(),
        freq='B',
    )
    market_holidays = [
        timestamp.strftime('%Y-%m-%d')
        for timestamp in calendar_weekdays.difference(plotted_dates)
    ]

    figure.add_trace(
        go.Candlestick(
            x=history_dates,
            open=context['open'],
            high=context['high'],
            low=context['low'],
            close=context['close'],
            name=f'历史行情（{len(context)}日）',
            increasing_line_color='#d85858',
            decreasing_line_color='#2f9d75',
            increasing_fillcolor='#d85858',
            decreasing_fillcolor='#2f9d75',
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=interval_df['close_p10'],
            mode='lines',
            line={'width': 0},
            hoverinfo='skip',
            showlegend=False,
            name='P10',
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=interval_df['close_p90'],
            mode='lines',
            line={'width': 0},
            fill='tonexty',
            fillcolor='rgba(202, 69, 69, 0.16)',
            name='80%预测区间（P10–P90）',
            hovertemplate='P90 %{y:.2f}<extra></extra>',
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=interval_df['close_p50'],
            mode='lines+markers',
            line={'color': '#bd3f3f', 'width': 2, 'dash': 'dot'},
            marker={'size': 5, 'color': '#bd3f3f'},
            name='预测中位数（P50）',
            hovertemplate='%{x|%Y-%m-%d}<br>P50 %{y:.2f}<extra></extra>',
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Candlestick(
            x=forecast_dates,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name=f'预测均值K线（{len(pred_df)}日）',
            increasing_line_color='#e27b45',
            decreasing_line_color='#3f78a8',
            increasing_fillcolor='#e27b45',
            decreasing_fillcolor='#3f78a8',
        ),
        row=1,
        col=1,
    )

    history_volume_colors = np.where(
        context['close'].to_numpy() >= context['open'].to_numpy(),
        'rgba(216, 88, 88, 0.62)',
        'rgba(47, 157, 117, 0.62)',
    )
    forecast_volume_colors = np.where(
        pred_df['close'].to_numpy() >= pred_df['open'].to_numpy(),
        'rgba(226, 123, 69, 0.72)',
        'rgba(63, 120, 168, 0.72)',
    )
    figure.add_trace(
        go.Bar(
            x=history_dates,
            y=context['volume'],
            marker_color=history_volume_colors,
            name='历史成交量',
            hovertemplate='%{x|%Y-%m-%d}<br>成交量 %{y:,.0f}<extra></extra>',
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=forecast_dates,
            y=pred_df['volume'],
            marker_color=forecast_volume_colors,
            name='预测成交量',
            hovertemplate='%{x|%Y-%m-%d}<br>预测量 %{y:,.0f}<extra></extra>',
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        template='plotly_white',
        height=640,
        showlegend=True,
        hovermode='x unified',
        margin={'l': 62, 'r': 26, 't': 50, 'b': 48},
        legend={'orientation': 'h', 'x': 0, 'y': 1.08},
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        bargap=0.18,
    )
    figure.update_xaxes(
        type='date',
        rangeslider_visible=False,
        rangebreaks=[
            {'bounds': ['sat', 'mon']},
            {'values': market_holidays},
        ],
    )
    figure.update_xaxes(title_text='交易日', row=2, col=1)
    figure.update_yaxes(title_text='价格', row=1, col=1)
    figure.update_yaxes(title_text='成交量', rangemode='tozero', row=2, col=1)
    return pio.to_json(figure, pretty=False)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Lightweight health endpoint for the reverse proxy and systemd checks."""
    return jsonify({
        'status': 'ok',
        'service': 'kronos-web',
        'backend': 'remote' if KRONOS_REMOTE_ONLY else KRONOS_INFERENCE_BACKEND,
        'model': KRONOS_MODEL_ID,
        'release': KRONOS_RELEASE_ID,
        'checkpoint': KRONOS_CHECKPOINT,
        'model_loaded': False if KRONOS_REMOTE_ONLY else predictor is not None,
    })


@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)


@app.route('/api/a-share/symbols')
def get_a_share_symbols():
    """Return the replaceable symbol-sector mapping used for autocomplete."""
    try:
        reference = load_sector_reference()
        result = [
            {
                'symbol': symbol,
                'latest_date': record['reference_date'],
                'sector_id': int(record['sector_id']),
                'sector_label': record['sector_label'],
            }
            for symbol, record in sorted(reference['symbols'].items())
        ]
        return jsonify({
            'symbols': result,
            'count': len(result),  # Backward compatibility for older web clients.
            'mapping_count': len(result),
            'mapping_reference_date': reference['date'].date().isoformat(),
            'sector_vocabulary_id': reference['vocabulary_id'],
            'training_panel_count': len(result),  # Backward compatibility.
            'market_scope': 'all-a-share',
        })
    except (FileNotFoundError, KeyError):
        # The production Oracle instance intentionally has no training dataset.
        return jsonify({
            'symbols': [], 'count': 0, 'mapping_count': 0,
            'mapping_reference_date': None, 'sector_vocabulary_id': None,
            'training_panel_count': 0,
            'market_scope': 'all-a-share',
            'message': '生产网关不保存训练面板，可直接输入股票代码',
        })
    except Exception as exc:
        return jsonify({'error': f'Failed to load A-share symbols: {exc}'}), 500


@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json() or {}
        requested_symbol = normalize_a_share_symbol(data.get('symbol'))
        if requested_symbol and re.fullmatch(r'(sh|sz|bj)\.\d{6}', requested_symbol):
            config = AVAILABLE_MODELS[KRONOS_MODEL_KEY]
            try:
                latest = latest_prediction_inputs(
                    requested_symbol,
                    int(config['default_lookback']),
                    int(config['default_pred_len']),
                )
                frame = latest['context']
                return jsonify({
                    'success': True,
                    'data_info': {
                        'rows': len(frame),
                        'columns': list(frame.columns),
                        'start_date': pd.Timestamp(frame['timestamps'].min()).isoformat(),
                        'end_date': pd.Timestamp(frame['timestamps'].max()).isoformat(),
                        'symbol': requested_symbol,
                        'name': latest.get('stock_name'),
                        'size_percentile': float(latest['size_percentile']),
                        'size_reference_date': latest['size_reference_date'].isoformat(),
                        'sector_id': int(latest['sector_id']),
                        'sector_label': latest['sector_label'],
                        'sector_reference_date': (
                            latest['sector_reference_date'].isoformat()
                            if latest['sector_reference_date'] is not None else None
                        ),
                        'latest_close': float(frame['close'].iloc[-1]),
                        'latest_volume': float(frame['volume'].iloc[-1]),
                        'remote_only': not latest.get('in_local_panel', False),
                        'data_source': latest['data_source'],
                    },
                    'message': (
                        f'{requested_symbol} 已刷新至 '
                        f"{pd.Timestamp(frame['timestamps'].iloc[-1]):%Y-%m-%d}"
                    ),
                })
            except Exception:
                return jsonify({
                    'success': True,
                    'data_info': {
                        'rows': 0, 'columns': [], 'start_date': None, 'end_date': None,
                        'symbol': requested_symbol, 'name': None,
                        'size_percentile': None, 'sector_id': None, 'sector_label': None,
                        'latest_close': None, 'latest_volume': None,
                        'remote_only': True,
                    },
                    'message': f'{requested_symbol} 将在预测时刷新最新行情',
                })
        df, error, _, symbol = resolve_request_data(data)
        if error:
            if requested_symbol and re.fullmatch(r'(sh|sz|bj)\.\d{6}', requested_symbol):
                return jsonify({
                    'success': True,
                    'data_info': {
                        'rows': 0,
                        'columns': [],
                        'start_date': None,
                        'end_date': None,
                        'symbol': requested_symbol,
                        'size_percentile': None,
                        'sector_id': None,
                        'sector_label': None,
                        'latest_close': None,
                        'latest_volume': None,
                        'remote_only': True,
                    },
                    'message': f'{requested_symbol} 将在预测时通过 BaoStock 获取最新数据',
                })
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        asof = pd.Timestamp(df['timestamps'].iloc[-1])
        local_size = size_condition_from_local_frame(df, asof=asof)
        local_sector = sector_condition_from_local_frame(df, asof=asof)
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df),
            'symbol': symbol,
            'name': None,
            'size_percentile': local_size['size_percentile'] if local_size else None,
            'sector_id': local_sector['sector_id'] if local_sector else None,
            'sector_label': local_sector['sector_label'] if local_sector else None,
            'latest_close': float(df['close'].iloc[-1]),
            'latest_volume': float(df['volume'].iloc[-1]) if 'volume' in df.columns else None,
            'remote_only': False,
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'{symbol} 历史数据已确认'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500


@app.route('/api/a-share/rankings', methods=['POST'])
def a_share_rankings():
    """Collect a stock pool locally and forecast it with the selected backend."""
    try:
        data = request.get_json() or {}
        raw_symbols = data.get('symbols')
        if not isinstance(raw_symbols, list):
            return jsonify({'error': 'symbols 必须是股票代码数组'}), 400
        symbols = list(dict.fromkeys(normalize_a_share_symbol(value) for value in raw_symbols))
        if not 2 <= len(symbols) <= 12 or any(not value for value in symbols):
            return jsonify({'error': '批量预测需要 2 到 12 个有效股票代码'}), 400
        config = AVAILABLE_MODELS[KRONOS_MODEL_KEY]
        lookback = int(config['default_lookback'])
        pred_len = int(config['default_pred_len'])
        sample_count = int(data.get('sample_count', config['default_sample_count']))
        temperature = float(data.get('temperature', config['default_temperature']))
        top_p = float(data.get('top_p', config['default_top_p']))
        backend = selected_backend(data.get('backend'))
        if backend not in ('local', 'remote'):
            return jsonify({'error': 'backend must be local or remote'}), 400
        batch = portfolio_ranking_batch(symbols, lookback, pred_len)
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        inference_results = []
        if backend == 'remote':
            items = []
            for record in batch['records']:
                items.append({
                    'id': record['symbol'],
                    'data': [
                        {'timestamp': pd.Timestamp(row['timestamps']).isoformat(), **{
                            column: float(row[column]) for column in feature_columns
                        }} for _, row in record['context'].iterrows()
                    ],
                    'future_timestamps': [
                        pd.Timestamp(value).isoformat() for value in record['future_dates']
                    ],
                    'sector_id': int(record['sector_id']),
                    'size_percentile': float(record['size_percentile']),
                })
            remote = call_remote_inference({
                'items': items, 'pred_len': pred_len, 'sample_count': sample_count,
                'temperature': temperature, 'top_p': top_p,
            }, endpoint='predict-batch', expected_key='results')
            inference_results = remote['results']
            model_device = remote.get('meta', {}).get('model_device', 'remote')
        else:
            model_device = None
            for record in batch['records']:
                predictions, _, _, close_samples, model_device = local_inference(
                    record['context'], record['future_dates'], pred_len, temperature,
                    top_p, sample_count, record['sector_id'], record['size_percentile'],
                )
                inference_results.append({
                    'id': record['symbol'],
                    'predictions': predictions,
                    'samples': {'close': close_samples.tolist()},
                })

        record_by_symbol = {record['symbol']: record for record in batch['records']}
        rankings = []
        for result in inference_results:
            record = record_by_symbol[result['id']]
            predictions = result['predictions']
            future_dates = pd.DatetimeIndex(record['future_dates'])
            pred_df = pd.DataFrame(predictions).set_index(future_dates)
            interval_df = pred_df[['close_p10', 'close_p50', 'close_p90']]
            latest_close = float(record['context']['close'].iloc[-1])
            close_samples = np.asarray(result['samples']['close'], dtype=np.float64)
            summary = forecast_return_summary(close_samples, latest_close)
            rankings.append({
                'symbol': result['id'],
                'name': (
                    None if record['stock_name'] and str(record['stock_name']).strip().lower() == result['id'].lower()
                    else record['stock_name']
                ),
                'latest_close': latest_close,
                'size_percentile': float(record['size_percentile']),
                'size_reference_date': record['size_reference_date'].isoformat(),
                'sector_id': int(record['sector_id']),
                'sector_label': record['sector_label'],
                'sector_reference_date': (
                    record['sector_reference_date'].isoformat()
                    if record['sector_reference_date'] is not None else None
                ),
                'data_source': record['data_source'], 'in_local_panel': record['in_local_panel'],
                'model_release': KRONOS_RELEASE_ID, 'model_checkpoint': KRONOS_CHECKPOINT,
                'latest_data_date': pd.Timestamp(record['context']['timestamps'].iloc[-1]).isoformat(),
                'forecast_start': future_dates[0].isoformat(),
                'forecast_end': future_dates[-1].isoformat(),
                'pred_len': pred_len, 'prediction_results': predictions,
                'chart': create_operational_chart(record['context'], pred_df, interval_df),
                'interval': {
                    'lower_quantile': 0.1, 'center_quantile': 0.5,
                    'upper_quantile': 0.9, 'sample_count': sample_count,
                },
                'final_close_p50': float(predictions[-1]['close_p50']),
                **summary,
            })
        rankings.sort(key=lambda row: row['predicted_return_p50'], reverse=True)
        batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        for ranking in rankings:
            saved = save_prediction_record({
                **ranking,
                'backend': backend,
                'batch_id': batch_id,
                'prediction_type': f'批量排名 · 最新 {lookback} 个交易日 → 未来 {pred_len} 个交易日',
                'prediction_params': {
                    'lookback': lookback, 'pred_len': pred_len,
                    'temperature': temperature, 'top_p': top_p,
                    'sample_count': sample_count, 'symbol': ranking['symbol'],
                },
            })
            ranking['record_id'] = saved['record_id']
        return jsonify({
            'success': True, 'as_of_date': batch['as_of_date'].isoformat(),
            'rankings': rankings, 'sample_count': sample_count,
            'backend': backend, 'model_device': model_device, 'batch_id': batch_id,
            'message': f'已完成 {len(rankings)} 只股票的{"远端" if backend == "remote" else "本地"}批量预测',
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'批量预测失败: {exc}'}), 500

@app.route('/api/predict-history', methods=['POST'])
def predict_history():
    """Perform prediction"""
    try:
        if KRONOS_REMOTE_ONLY:
            return jsonify({'error': '生产网关仅支持 Modal 最新预测，不提供本地历史回测'}), 400
        data = request.get_json() or {}
        defaults = current_model_config or {}
        lookback = int(data.get('lookback', defaults.get('default_lookback', 400)))
        pred_len = int(data.get('pred_len', defaults.get('default_pred_len', 120)))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        df, error, source_path, symbol = resolve_request_data(data)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback + pred_len:
            return jsonify({'error': f'Insufficient data length, need at least {lookback + pred_len} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Keep the same six inputs used during training: OHLCV and amount.
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                if 'amount' in df.columns:
                    required_cols.append('amount')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Preserve the original comparison mode when no explicit date is selected.
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"

                context_frame = time_range_df if start_date else df
                signal_date = pd.Timestamp(context_frame.iloc[lookback - 1]['timestamps'])
                size_condition = size_condition_from_local_frame(context_frame, asof=signal_date)
                sector_condition = sector_condition_from_local_frame(context_frame, asof=signal_date)
                if size_condition is None:
                    return jsonify({'error': 'Beta V1.2 requires a point-in-time size_percentile'}), 400
                if sector_condition is None:
                    return jsonify({'error': 'Beta V1.2 requires a point-in-time sector label'}), 400
                size_percentile = size_condition['size_percentile']
                sector_id = sector_condition['sector_id']
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                _seed_inference(KRONOS_INFERENCE_SEED)
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    sector_id=sector_id,
                    size_bucket=None,
                    size_percentile=size_percentile,
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Backtest mode already has the exact future trading-day index.
        if len(y_timestamp) >= pred_len:
            future_timestamps = pd.DatetimeIndex(pd.to_datetime(y_timestamp.iloc[:pred_len]))
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=source_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest',
                    'symbol': symbol,
                    'size_percentile': size_percentile,
                    'sector_id': sector_id,
                    'sector_label': sector_condition['sector_label'],
                    'model_key': current_model_key,
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'symbol': symbol,
            'size_percentile': size_percentile,
            'sector_id': sector_id,
            'sector_label': sector_condition['sector_label'],
            'model_key': current_model_key,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_latest():
    """Refresh the selected stock and forecast the next trading days."""
    try:
        data = request.get_json() or {}
        backend = selected_backend(data.get('backend'))
        symbol = normalize_a_share_symbol(data.get('symbol'))
        if not symbol:
            return jsonify({'error': '请输入股票代码'}), 400

        config = AVAILABLE_MODELS[KRONOS_MODEL_KEY]
        lookback = int(config['default_lookback'])
        pred_len = int(config['default_pred_len'])
        temperature = float(data.get('temperature', config['default_temperature']))
        top_p = float(data.get('top_p', config['default_top_p']))
        sample_count = int(data.get('sample_count', config['default_sample_count']))
        if not 5 <= sample_count <= 50:
            return jsonify({'error': 'sample_count 必须在 5 到 50 之间'}), 400

        inputs = latest_prediction_inputs(symbol, lookback, pred_len)
        context = inputs['context']
        future_dates = inputs['future_dates']
        size_percentile = inputs['size_percentile']
        sector_id = inputs['sector_id']

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        x_df = context[feature_columns]
        y_timestamp = pd.Series(
            pd.to_datetime(future_dates).to_numpy(),
            name='timestamps',
        )
        remote_payload = {
            'data': [
                {'timestamp': pd.Timestamp(row['timestamps']).isoformat(), **{
                    column: float(row[column]) for column in feature_columns
                }}
                for _, row in context.iterrows()
            ],
            'future_timestamps': [pd.Timestamp(value).isoformat() for value in y_timestamp],
            'pred_len': pred_len,
            'temperature': temperature,
            'top_p': top_p,
            'sample_count': sample_count,
            'sector_id': sector_id,
            'size_percentile': size_percentile,
        }
        if backend == 'remote':
            remote_result = call_remote_inference(remote_payload)
            prediction_results = remote_result['predictions']
            if len(prediction_results) != pred_len:
                raise RuntimeError(f'Unexpected prediction count from Modal: {len(prediction_results)}')
            pred_df = pd.DataFrame(prediction_results).set_index(pd.DatetimeIndex(y_timestamp))
            interval_df = pred_df[['close_p10', 'close_p50', 'close_p90']]
            close_samples = np.asarray(remote_result.get('samples', {}).get('close', []), dtype=np.float64)
            if close_samples.shape != (sample_count, pred_len):
                raise RuntimeError(f'Unexpected close sample shape from Modal: {close_samples.shape}')
            model_device = remote_result.get('meta', {}).get('model_device', 'remote')
        elif backend == 'local':
            prediction_results, pred_df, interval_df, close_samples, model_device = local_inference(
                context, future_dates, pred_len, temperature, top_p, sample_count,
                sector_id, size_percentile,
            )
        else:
            raise RuntimeError('backend must be local or remote')

        chart_json = create_operational_chart(
            context,
            pred_df,
            interval_df,
        )
        latest_date = pd.Timestamp(context['timestamps'].iloc[-1])
        latest_close = float(context['close'].iloc[-1])
        forecast_summary = forecast_return_summary(close_samples, latest_close)
        prediction_type = f'最新 {lookback} 个交易日 → 未来 {pred_len} 个交易日'
        return jsonify({
            'success': True,
            'symbol': symbol,
            'name': inputs['stock_name'],
            'model_key': current_model_key or KRONOS_MODEL_KEY,
            'model_release': KRONOS_RELEASE_ID,
            'model_checkpoint': KRONOS_CHECKPOINT,
            'model_device': model_device,
            'prediction_type': prediction_type,
            'lookback': lookback,
            'pred_len': pred_len,
            'latest_data_date': latest_date.isoformat(),
            'latest_close': latest_close,
            **forecast_summary,
            'forecast_start': pd.Timestamp(y_timestamp.iloc[0]).isoformat(),
            'forecast_end': pd.Timestamp(y_timestamp.iloc[-1]).isoformat(),
            'size_percentile': size_percentile,
            'size_percentile_asof': inputs['size_percentile_asof'].isoformat(),
            'size_reference_date': inputs['size_reference_date'].isoformat(),
            'size_source': inputs['size_source'],
            'estimated_market_cap': inputs['estimated_market_cap'],
            'sector_id': sector_id,
            'sector_label': inputs['sector_label'],
            'sector_reference_date': (
                inputs['sector_reference_date'].isoformat()
                if inputs['sector_reference_date'] is not None else None
            ),
            'sector_source': inputs['sector_source'],
            'in_local_panel': inputs['in_local_panel'],
            'data_source': inputs['data_source'],
            'calendar_source': inputs['calendar_source'],
            'refresh_error': inputs['refresh_error'],
            'chart': chart_json,
            'prediction_results': prediction_results,
            'interval': {
                'lower_quantile': 0.1,
                'center_quantile': 0.5,
                'upper_quantile': 0.9,
                'sample_count': sample_count,
            },
            'actual_data': [],
            'has_comparison': False,
            'record_id': save_prediction_record({
                'symbol': symbol,
                'name': inputs['stock_name'],
                'backend': backend,
                'prediction_type': prediction_type,
                'prediction_params': {
                    'lookback': lookback, 'pred_len': pred_len,
                    'temperature': temperature, 'top_p': top_p,
                    'sample_count': sample_count, 'symbol': symbol,
                },
                'latest_close': latest_close,
                'predicted_return_p50': forecast_summary['predicted_return_p50'],
                'chart': chart_json,
                'prediction_results': prediction_results,
                'actual_data': [],
                'lookback': lookback, 'pred_len': pred_len,
                'latest_data_date': latest_date.isoformat(),
                'forecast_start': pd.Timestamp(y_timestamp.iloc[0]).isoformat(),
                'forecast_end': pd.Timestamp(y_timestamp.iloc[-1]).isoformat(),
                'model_key': KRONOS_MODEL_KEY,
                'model_release': KRONOS_RELEASE_ID,
                'model_checkpoint': KRONOS_CHECKPOINT,
                'size_percentile': size_percentile,
                'size_percentile_asof': inputs['size_percentile_asof'].isoformat(),
                'size_reference_date': inputs['size_reference_date'].isoformat(),
                'sector_id': sector_id,
                'sector_label': inputs['sector_label'],
                'sector_reference_date': (
                    inputs['sector_reference_date'].isoformat()
                    if inputs['sector_reference_date'] is not None else None
                ),
                'data_source': inputs['data_source'],
                'in_local_panel': inputs['in_local_panel'],
                'interval': {'lower_quantile': 0.1, 'center_quantile': 0.5,
                             'upper_quantile': 0.9, 'sample_count': sample_count},
                'timing_signal': forecast_summary['timing_signal'],
            })['record_id'],
            'message': f'{symbol} 已使用截至 {latest_date:%Y-%m-%d} 的最新数据完成预测',
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'预测失败: {exc}'}), 500


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load the Beta V1.2 candidate on the selected inference backend."""
    try:
        requested_backend = selected_backend((request.get_json(silent=True) or {}).get('backend'))
        if KRONOS_REMOTE_ONLY and requested_backend != 'remote':
            return jsonify({'error': '本部署仅支持 Modal 远端推理'}), 400
        if requested_backend == 'remote':
            return jsonify({
                'success': True,
                'message': 'Modal 远端模式已就绪，将在预测时按需启动',
                'model_info': {
                    'name': KRONOS_MODEL_ID,
                    'params': AVAILABLE_MODELS[KRONOS_MODEL_KEY]['params'],
                    'device': 'modal',
                    'backend': 'remote',
                    'release_id': KRONOS_RELEASE_ID,
                    'checkpoint': KRONOS_CHECKPOINT,
                    'num_sectors': KRONOS_NUM_SECTORS,
                    'use_size_percentile': True,
                },
            })
        device = ensure_model_loaded()
        model_config = AVAILABLE_MODELS[KRONOS_MODEL_KEY]
        return jsonify({
            'success': True,
            'message': f'模型已就绪，运行设备：{device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'device': str(next(predictor.model.parameters()).device),
                'context_length': model_config['context_length'],
                'description': model_config['description'],
                'backend': 'local',
                'release_id': model_config['release_id'],
                'checkpoint': model_config['checkpoint'],
                'num_sectors': model_config['num_sectors'],
                'use_size_percentile': model_config['use_size_percentile'],
                'num_size_buckets': model_config.get('num_size_buckets', 0),
                'default_lookback': model_config.get('default_lookback', 400),
                'default_pred_len': model_config.get('default_pred_len', 120),
            }
        })
        
    except Exception as exc:
        return jsonify({'error': f'模型加载失败: {exc}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': (
            {**AVAILABLE_MODELS, KRONOS_MODEL_KEY: {
                **AVAILABLE_MODELS[KRONOS_MODEL_KEY],
                'model_id': KRONOS_MODEL_ID,
                'tokenizer_id': f'{KRONOS_MODEL_ID}/tokenizer',
            }} if KRONOS_REMOTE_ONLY else AVAILABLE_MODELS
        ),
        'model_available': MODEL_AVAILABLE and not KRONOS_REMOTE_ONLY,
        'remote_only': KRONOS_REMOTE_ONLY,
        'recommended_model': KRONOS_MODEL_KEY,
        'recommended_device': 'modal' if KRONOS_REMOTE_ONLY else automatic_device(),
        'devices': {
            'cpu': True,
            'mps': bool(torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
            'cuda': bool(torch is not None and torch.cuda.is_available()),
        },
    })

@app.route('/api/model-status')
def get_model_status():
    """Return the local or remote runtime status without loading a model."""
    if KRONOS_REMOTE_ONLY:
        return jsonify({
            'available': True,
            'loaded': True,
            'remote_only': True,
            'message': 'Modal 远端推理已就绪',
            'current_model': {
                'key': KRONOS_MODEL_ID,
                'device': 'modal',
                'release_id': KRONOS_RELEASE_ID,
                'checkpoint': KRONOS_CHECKPOINT,
            },
        })
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': '模型已就绪',
                'current_model': {
                    'key': current_model_key,
                    'name': (current_model_config or {}).get('name', predictor.model.__class__.__name__),
                    'device': str(next(predictor.model.parameters()).device),
                    'release_id': (current_model_config or {}).get('release_id'),
                    'checkpoint': (current_model_config or {}).get('checkpoint'),
                    'num_sectors': (current_model_config or {}).get('num_sectors', 0),
                    'use_size_percentile': (current_model_config or {}).get('use_size_percentile', False),
                    'num_size_buckets': (current_model_config or {}).get('num_size_buckets', 0),
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': '模型等待自动加载',
                'recommended_device': automatic_device(),
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=7070)
