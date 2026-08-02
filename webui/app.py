import os
import pickle
import re
import threading
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    bs = None
    BAOSTOCK_AVAILABLE = False

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    from model.kronos import auto_regressive_inference
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_SHARE_DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'a_share', 'processed_datasets')
A_SHARE_RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'a_share', 'a_share_daily.csv')
A_SHARE_SIZE_REFERENCE_PATH = os.path.join(PROJECT_ROOT, 'webui', 'size_reference.json')
A_SHARE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    'outputs',
    'models',
    'a_share_size_kronos_base_earlystop50',
    'checkpoints',
    'best_model',
)

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
ranking_cache = None
size_reference_cache = None
size_reference_lock = threading.Lock()

# Available model configurations
AVAILABLE_MODELS = {
    'a-share-size-kronos-base': {
        'name': 'A-share Size Kronos-base',
        'model_id': A_SHARE_MODEL_PATH,
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'A-share daily model with point-in-time market-cap conditioning',
        'num_size_buckets': 10,
        'use_size_percentile': False,
        'default_lookback': 90,
        'default_pred_len': 10,
        'default_temperature': 0.6,
        'default_top_p': 0.9,
        'default_sample_count': 20,
        'model_kwargs': {
            'num_sectors': 0,
            'num_size_buckets': 10,
            'context_layer': 10,
            'use_size_percentile': False,
            'size_mlp_hidden_dim': 64,
        },
        'local': True,
    }
}


def automatic_device():
    """Choose the accelerator without exposing device selection in the UI."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
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

        model_key = 'a-share-size-kronos-base'
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


def query_latest_daily_data(symbol, lookback, pred_len):
    """Fetch a fresh adjusted context and the next exchange trading dates."""
    if not BAOSTOCK_AVAILABLE:
        raise RuntimeError('BaoStock is not installed')

    today = pd.Timestamp.now().normalize()
    history_start = today - pd.Timedelta(days=max(365, lookback * 4))
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
            if len(history_rows) < lookback:
                raise RuntimeError(
                    f'BaoStock returned {len(history_rows)} rows; {lookback} are required'
                )

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
            history = history.tail(lookback).reset_index(drop=True)

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


def load_latest_size_reference():
    """Load the latest available cross-section used to size unseen stocks."""
    global size_reference_cache
    if size_reference_cache is not None:
        return size_reference_cache

    with size_reference_lock:
        if size_reference_cache is not None:
            return size_reference_cache
        if os.path.exists(A_SHARE_SIZE_REFERENCE_PATH):
            with open(A_SHARE_SIZE_REFERENCE_PATH) as handle:
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


def size_condition_from_remote_context(context):
    """Estimate float market cap and map it to the training cross-section."""
    ordered = context.sort_values('timestamps').copy()
    for column in ('close', 'volume', 'turn'):
        ordered[column] = pd.to_numeric(ordered[column], errors='coerce')
    usable = ordered[
        np.isfinite(ordered['close'])
        & np.isfinite(ordered['volume'])
        & np.isfinite(ordered['turn'])
        & (ordered['close'] > 0)
        & (ordered['volume'] > 0)
        & (ordered['turn'] > 0)
    ]
    if usable.empty:
        raise ValueError('BaoStock history has no usable turnover row for market-cap sizing')
    latest = usable.iloc[-1]
    close = float(latest['close'])
    volume = float(latest['volume'])
    turnover = float(latest['turn'])
    market_cap = close * volume / (turnover / 100.0)
    if not np.isfinite(market_cap) or market_cap <= 0:
        raise ValueError('Unable to estimate a positive float market cap from BaoStock')

    reference = load_latest_size_reference()
    percentile = float(
        np.searchsorted(reference['market_caps'], market_cap, side='right')
        / reference['count']
    )
    percentile = float(np.clip(percentile, 0.0, 1.0))
    size_bucket = min(int(np.floor(percentile * 10)), 9)
    return {
        'size_bucket': size_bucket,
        'size_percentile': percentile,
        'size_bucket_asof': pd.Timestamp(latest['timestamps']),
        'size_reference_date': reference['date'],
        'size_source': 'baostock_proxy_vs_local_cross_section',
        'estimated_market_cap': float(market_cap),
    }


def size_condition_from_local_frame(local_frame):
    if local_frame is None or 'size_bucket' not in local_frame.columns:
        return None
    rows = local_frame.dropna(subset=['size_bucket'])
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    size_bucket = int(latest['size_bucket'])
    percentile = latest.get('size_percentile', np.nan)
    if not np.isfinite(percentile):
        percentile = (size_bucket + 0.5) / 10.0
    asof = pd.Timestamp(latest['timestamps'])
    return {
        'size_bucket': size_bucket,
        'size_percentile': float(np.clip(percentile, 0.0, 1.0)),
        'size_bucket_asof': asof,
        'size_reference_date': asof,
        'size_source': 'local_panel',
        'estimated_market_cap': None,
    }


def latest_prediction_inputs(symbol, lookback, pred_len):
    """Refresh a stock and derive size context even when it is outside the panel."""
    symbol = normalize_a_share_symbol(symbol)
    local_frame, local_error = get_a_share_symbol_frame(symbol)
    local_size = size_condition_from_local_frame(local_frame)

    try:
        context, future_dates = query_latest_daily_data(symbol, lookback, pred_len)
        try:
            size_context = size_condition_from_remote_context(context)
        except Exception:
            if local_size is None:
                raise
            size_context = local_size
        data_source = 'baostock'
        calendar_source = 'baostock'
        refresh_error = None
    except Exception as exc:
        if local_frame is None or local_size is None:
            raise ValueError(
                f'{symbol} 不在本地训练面板中，且 BaoStock 最新数据无法用于预测：{exc}'
            ) from exc
        context = local_frame.tail(lookback).copy().reset_index(drop=True)
        last_date = pd.Timestamp(context['timestamps'].iloc[-1])
        future_dates = pd.Series(
            pd.bdate_range(last_date + pd.Timedelta(days=1), periods=pred_len),
            name='timestamps',
        )
        size_context = local_size
        data_source = 'local_cache'
        calendar_source = 'business_day_fallback'
        refresh_error = str(exc)

    if len(context) < lookback:
        raise ValueError(f'{symbol} has only {len(context)} rows; {lookback} are required')
    return {
        'context': context.tail(lookback).reset_index(drop=True),
        'future_dates': future_dates,
        **size_context,
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


def kronos_time_features(index):
    index = pd.DatetimeIndex(index)
    return np.column_stack([
        index.minute,
        index.hour,
        index.weekday,
        index.day,
        index.month,
    ]).astype(np.float32)


def portfolio_ranking_batch(symbols, lookback, pred_len):
    """Build a same-date batch for a user-provided A-share universe."""
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    frames = {}
    for symbol in symbols:
        frame, error = get_a_share_symbol_frame(symbol)
        if error:
            raise ValueError(error)
        frame = frame.set_index('timestamps').sort_index()
        if len(frame) < lookback:
            raise ValueError(f'{symbol} 只有 {len(frame)} 个交易日，至少需要 {lookback} 个')
        frames[symbol] = frame

    common_dates = None
    for frame in frames.values():
        dates = pd.DatetimeIndex(frame.index)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates is None or common_dates.empty:
        raise ValueError('股票池没有共同交易日期，无法公平排序')
    as_of_date = pd.Timestamp(common_dates.max())

    records = []
    for symbol, frame in frames.items():
        context = frame.loc[:as_of_date].tail(lookback)
        required = feature_columns + ['size_bucket']
        if AVAILABLE_MODELS['a-share-size-kronos-base'].get('use_size_percentile'):
            required.append('size_percentile')
        if context[required].isnull().values.any():
            raise ValueError(f'{symbol} 在共同日期前存在缺失行情或市值层')
        size_bucket = int(context['size_bucket'].iloc[-1])
        size_percentile = context.iloc[-1].get('size_percentile', np.nan)
        if not np.isfinite(size_percentile):
            size_percentile = (size_bucket + 0.5) / 10.0
        records.append({
            'symbol': symbol,
            'context': context,
            'size_bucket': size_bucket,
            'size_percentile': float(size_percentile),
            'latest_close': float(context['close'].iloc[-1]),
        })

    future_dates = pd.bdate_range(
        as_of_date + pd.Timedelta(days=1), periods=pred_len
    )
    normalized = []
    x_stamps = []
    means = []
    stds = []
    size_buckets = []
    size_percentiles = []
    for item in records:
        context = item['context']
        values = context[feature_columns].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        normalized.append(np.clip((values - mean) / (std + 1e-5), -5, 5))
        x_stamps.append(kronos_time_features(context.index))
        means.append(mean)
        stds.append(std)
        size_buckets.append(item['size_bucket'])
        size_percentiles.append(item['size_percentile'])

    return {
        'records': records,
        'as_of_date': as_of_date,
        'future_dates': future_dates,
        'feature_columns': feature_columns,
        'x': np.stack(normalized).astype(np.float32),
        'x_stamp': np.stack(x_stamps).astype(np.float32),
        'y_stamp': np.repeat(
            kronos_time_features(future_dates)[None, :, :], len(records), axis=0
        ),
        'means': np.stack(means).astype(np.float32),
        'stds': np.stack(stds).astype(np.float32),
        'size_buckets': np.asarray(size_buckets, dtype=np.int64),
        'size_percentiles': np.asarray(size_percentiles, dtype=np.float32),
    }


def generate_portfolio_ranking(symbols, sample_count=3):
    """Rank a user-provided stock pool by median forecast return."""
    global ranking_cache
    ensure_model_loaded()
    batch = portfolio_ranking_batch(
        symbols,
        AVAILABLE_MODELS['a-share-size-kronos-base']['default_lookback'],
        AVAILABLE_MODELS['a-share-size-kronos-base']['default_pred_len'],
    )
    cache_key = (
        batch['as_of_date'].isoformat(), tuple(symbols), sample_count, current_model_key
    )
    if ranking_cache and ranking_cache.get('cache_key') == cache_key:
        return ranking_cache['payload']

    device = next(predictor.model.parameters()).device
    close_index = batch['feature_columns'].index('close')
    final_close_samples = []
    with inference_lock, torch.no_grad():
        for sample_index in range(sample_count):
            torch.manual_seed(20260801 + sample_index)
            np.random.seed(20260801 + sample_index)
            predictions = auto_regressive_inference(
                tokenizer,
                model,
                torch.as_tensor(batch['x'], device=device),
                torch.as_tensor(batch['x_stamp'], device=device),
                torch.as_tensor(batch['y_stamp'], device=device),
                max_context=AVAILABLE_MODELS['a-share-size-kronos-base']['context_length'],
                pred_len=AVAILABLE_MODELS['a-share-size-kronos-base']['default_pred_len'],
                clip=5,
                T=AVAILABLE_MODELS['a-share-size-kronos-base']['default_temperature'],
                top_k=0,
                top_p=AVAILABLE_MODELS['a-share-size-kronos-base']['default_top_p'],
                sample_count=1,
                verbose=False,
                size_bucket=torch.as_tensor(batch['size_buckets'], device=device),
                size_percentile=(
                    torch.as_tensor(batch['size_percentiles'], device=device)
                    if AVAILABLE_MODELS['a-share-size-kronos-base'].get('use_size_percentile')
                    else None
                ),
            )
            forecast = predictions[:, -AVAILABLE_MODELS['a-share-size-kronos-base']['default_pred_len']:, :]
            final_close = (
                forecast[:, -1, close_index]
                * (batch['stds'][:, close_index] + 1e-5)
                + batch['means'][:, close_index]
            )
            final_close_samples.append(final_close)
        if device.type == 'mps':
            torch.mps.synchronize()

    final_close_samples = np.stack(final_close_samples)
    rankings = []
    for item_index, item in enumerate(batch['records']):
        closes = final_close_samples[:, item_index]
        returns = closes / item['latest_close'] - 1
        rankings.append({
            'symbol': item['symbol'],
            'size_bucket': item['size_bucket'],
            'size_percentile': item['size_percentile'],
            'latest_close': item['latest_close'],
            'predicted_close_p50': float(np.median(closes)),
            'predicted_return_p50': float(np.median(returns)),
            'positive_path_rate': float(np.mean(returns > 0)),
        })
    rankings.sort(
        key=lambda item: (-item['predicted_return_p50'], item['symbol'])
    )
    for rank, item in enumerate(rankings, start=1):
        item['rank'] = rank
        if rank <= 8 and item['predicted_return_p50'] > 0:
            item['signal'] = 'candidate'
        elif item['predicted_return_p50'] > 0:
            item['signal'] = 'positive'
        else:
            item['signal'] = 'negative'

    payload = {
        'success': True,
        'as_of_date': batch['as_of_date'].isoformat(),
        'forecast_end': batch['future_dates'][-1].isoformat(),
        'universe_size': len(symbols),
        'sample_count': sample_count,
        'rankings': rankings,
        'message': f'已完成自选股票池 {len(symbols)} 只股票的横截面排序',
    }
    ranking_cache = {'cache_key': cache_key, 'payload': payload}
    return payload


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
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
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
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


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
        rangebreaks=[{'bounds': ['sat', 'mon']}],
    )
    figure.update_xaxes(title_text='交易日', row=2, col=1)
    figure.update_yaxes(title_text='价格', row=1, col=1)
    figure.update_yaxes(title_text='成交量', rangemode='tozero', row=2, col=1)
    return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)


@app.route('/api/a-share/symbols')
def get_a_share_symbols():
    """Return built-in A-share symbols and their latest size buckets."""
    try:
        splits = load_a_share_splits()
        symbols = sorted(set(splits['train']) | set(splits['val']))
        result = []
        for symbol in symbols:
            rows = splits['val'].get(symbol)
            if rows is None or rows.empty:
                rows = splits['train'].get(symbol)
            latest = rows.iloc[-1]
            result.append({
                'symbol': symbol,
                'latest_date': pd.Timestamp(rows.index[-1]).strftime('%Y-%m-%d'),
                'size_bucket': int(latest['size_bucket']),
                'close': float(latest['close']),
            })
        return jsonify({'symbols': result, 'count': len(result)})
    except Exception as exc:
        return jsonify({'error': f'Failed to load A-share symbols: {exc}'}), 500


@app.route('/api/a-share/rankings', methods=['POST'])
def rank_a_share_portfolio():
    """Rank a user-provided stock pool on one common point-in-time date."""
    try:
        data = request.get_json() or {}
        raw_symbols = data.get('symbols', [])
        if isinstance(raw_symbols, str):
            raw_symbols = re.split(r'[\s,，;；]+', raw_symbols.strip())
        if not isinstance(raw_symbols, list):
            return jsonify({'error': '股票池格式无效'}), 400

        symbols = []
        for raw_symbol in raw_symbols:
            symbol = normalize_a_share_symbol(raw_symbol)
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        if len(symbols) < 2:
            return jsonify({'error': '股票池至少需要 2 只股票'}), 400
        if len(symbols) > 64:
            return jsonify({'error': '股票池最多支持 64 只股票'}), 400
        symbols.sort()

        sample_count = int(data.get('sample_count', 3))
        if not 1 <= sample_count <= 5:
            return jsonify({'error': '排序路径数必须在 1 到 5 之间'}), 400
        return jsonify(generate_portfolio_ranking(symbols, sample_count=sample_count))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'股票池排序失败: {exc}'}), 500

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json() or {}
        df, error, _, symbol = resolve_request_data(data)
        if error:
            requested_symbol = normalize_a_share_symbol(data.get('symbol'))
            if requested_symbol and re.fullmatch(r'(sh|sz|bj)\.\d{6}', requested_symbol):
                return jsonify({
                    'success': True,
                    'data_info': {
                        'rows': 0,
                        'columns': [],
                        'start_date': None,
                        'end_date': None,
                        'symbol': requested_symbol,
                        'size_bucket': None,
                        'size_percentile': None,
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
            'size_bucket': int(df['size_bucket'].iloc[-1]) if 'size_bucket' in df.columns else None,
            'size_percentile': (
                float(df['size_percentile'].iloc[-1])
                if 'size_percentile' in df.columns else None
            ),
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

@app.route('/api/predict-history', methods=['POST'])
def predict_history():
    """Perform prediction"""
    try:
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

                size_bucket = data.get('size_bucket')
                context_frame = time_range_df if start_date else df
                if size_bucket is None and 'size_bucket' in context_frame.columns:
                    size_bucket = int(context_frame.iloc[lookback - 1]['size_bucket'])

                bucket_count = int((current_model_config or {}).get('num_size_buckets', 0))
                if bucket_count > 0:
                    if size_bucket is None:
                        return jsonify({'error': 'The selected model requires a market-cap bucket'}), 400
                    size_bucket = int(size_bucket)
                    if not 0 <= size_bucket < bucket_count:
                        return jsonify({'error': f'Market-cap bucket must be between 0 and {bucket_count - 1}'}), 400
                else:
                    size_bucket = None
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    size_bucket=size_bucket,
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
                    'size_bucket': size_bucket,
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
            'size_bucket': size_bucket,
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
        symbol = normalize_a_share_symbol(data.get('symbol'))
        if not symbol:
            return jsonify({'error': '请输入股票代码'}), 400

        config = AVAILABLE_MODELS['a-share-size-kronos-base']
        lookback = int(config['default_lookback'])
        pred_len = int(config['default_pred_len'])
        temperature = float(data.get('temperature', config['default_temperature']))
        top_p = float(data.get('top_p', config['default_top_p']))
        sample_count = int(data.get('sample_count', config['default_sample_count']))
        if not 5 <= sample_count <= 50:
            return jsonify({'error': 'sample_count 必须在 5 到 50 之间'}), 400

        ensure_model_loaded()
        inputs = latest_prediction_inputs(symbol, lookback, pred_len)
        context = inputs['context']
        future_dates = inputs['future_dates']
        size_bucket = inputs['size_bucket']
        size_percentile = inputs['size_percentile']

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        x_df = context[feature_columns]
        x_timestamp = pd.Series(
            pd.to_datetime(context['timestamps']).to_numpy(),
            name='timestamps',
        )
        y_timestamp = pd.Series(
            pd.to_datetime(future_dates).to_numpy(),
            name='timestamps',
        )
        with inference_lock:
            prediction_samples = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
                size_bucket=size_bucket,
                size_percentile=(
                    size_percentile if config.get('use_size_percentile') else None
                ),
                return_samples=True,
            )

        feature_count = len(feature_columns)
        if prediction_samples.shape != (sample_count, pred_len, feature_count):
            raise RuntimeError(
                f'Unexpected prediction sample shape: {prediction_samples.shape}'
            )
        mean_prediction = prediction_samples.mean(axis=0)
        pred_df = pd.DataFrame(
            mean_prediction,
            columns=feature_columns,
            index=pd.DatetimeIndex(y_timestamp),
        )
        pred_df['high'] = pred_df[['open', 'high', 'low', 'close']].max(axis=1)
        pred_df['low'] = pred_df[['open', 'high', 'low', 'close']].min(axis=1)
        pred_df['volume'] = pred_df['volume'].clip(lower=0)
        pred_df['amount'] = pred_df['amount'].clip(lower=0)

        close_samples = prediction_samples[:, :, feature_columns.index('close')]
        close_quantiles = np.quantile(close_samples, [0.1, 0.5, 0.9], axis=0)
        interval_df = pd.DataFrame(
            {
                'close_p10': close_quantiles[0],
                'close_p50': close_quantiles[1],
                'close_p90': close_quantiles[2],
            },
            index=pd.DatetimeIndex(y_timestamp),
        )

        prediction_results = []
        for row_index, (timestamp, (_, row)) in enumerate(zip(y_timestamp, pred_df.iterrows())):
            prediction_results.append({
                'timestamp': pd.Timestamp(timestamp).isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'amount': float(row['amount']),
                'close_p10': float(interval_df.iloc[row_index]['close_p10']),
                'close_p50': float(interval_df.iloc[row_index]['close_p50']),
                'close_p90': float(interval_df.iloc[row_index]['close_p90']),
            })

        chart_json = create_operational_chart(
            context,
            pred_df,
            interval_df,
        )
        latest_date = pd.Timestamp(context['timestamps'].iloc[-1])
        latest_close = float(context['close'].iloc[-1])
        prediction_type = f'最新 {lookback} 个交易日 → 未来 {pred_len} 个交易日'
        model_device = str(next(predictor.model.parameters()).device)

        save_prediction_results(
            file_path=f'a-share:{symbol}',
            prediction_type=prediction_type,
            prediction_results=prediction_results,
            actual_data=[],
            input_data=x_df,
            prediction_params={
                'lookback': lookback,
                'pred_len': pred_len,
                'temperature': temperature,
                'top_p': top_p,
                'sample_count': sample_count,
                'symbol': symbol,
                'size_bucket': size_bucket,
                'size_percentile': size_percentile,
                'size_bucket_asof': inputs['size_bucket_asof'].strftime('%Y-%m-%d'),
                'size_reference_date': inputs['size_reference_date'].strftime('%Y-%m-%d'),
                'size_source': inputs['size_source'],
                'estimated_market_cap': inputs['estimated_market_cap'],
                'in_local_panel': inputs['in_local_panel'],
                'model_key': current_model_key,
                'data_source': inputs['data_source'],
            },
        )

        return jsonify({
            'success': True,
            'symbol': symbol,
            'model_key': current_model_key,
            'model_device': model_device,
            'prediction_type': prediction_type,
            'lookback': lookback,
            'pred_len': pred_len,
            'latest_data_date': latest_date.isoformat(),
            'latest_close': latest_close,
            'forecast_start': pd.Timestamp(y_timestamp.iloc[0]).isoformat(),
            'forecast_end': pd.Timestamp(y_timestamp.iloc[-1]).isoformat(),
            'size_bucket': size_bucket,
            'size_percentile': size_percentile,
            'size_bucket_asof': inputs['size_bucket_asof'].isoformat(),
            'size_reference_date': inputs['size_reference_date'].isoformat(),
            'size_source': inputs['size_source'],
            'estimated_market_cap': inputs['estimated_market_cap'],
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
            'message': f'{symbol} 已使用截至 {latest_date:%Y-%m-%d} 的最新数据完成预测',
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'预测失败: {exc}'}), 500


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load the production model on the automatically selected device."""
    try:
        device = ensure_model_loaded()
        model_config = AVAILABLE_MODELS['a-share-size-kronos-base']
        return jsonify({
            'success': True,
            'message': f'模型已就绪，运行设备：{device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'device': str(next(predictor.model.parameters()).device),
                'context_length': model_config['context_length'],
                'description': model_config['description'],
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
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE,
        'recommended_model': 'a-share-size-kronos-base',
        'recommended_device': automatic_device(),
        'devices': {
            'cpu': True,
            'mps': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
            'cuda': bool(torch.cuda.is_available()),
        },
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
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
