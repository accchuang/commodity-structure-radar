"""
fetcher_akshare.py — After-market institutional ranking data pipeline.

Fetches futures position rankings via Sina Finance API (akshare),
cleans and normalizes the data, stores to Parquet, and synthesizes daily bias.

Data source: Sina Finance futures position rankings
  https://vip.stock.finance.sina.com.cn/q/view/vFutures_Positions_cjcc.php

Usage:
    python src/fetcher_akshare.py                     # fetch latest data
    python src/fetcher_akshare.py --date 20260710     # fetch specific date
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

# 确保项目根目录在 sys.path 中，使 `from src.xxx` 导入生效
# (无论以 `python src/fetcher_akshare.py` 还是 `python -m src.fetcher_akshare` 运行)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from src.config import (
    AKSHARE_REQUEST_DELAY,
    BIAS_DATA_DIR,
    BIAS_FILE_NAME,
    MONITORED_SYMBOLS,
    PRICE_DATA_DIR,
    RAW_DATA_DIR,
)
from src.indicators import (
    calc_concentration_ratio,
    calc_daily_bias,
    calc_net_position_momentum,
)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Fetch position data from Sina Finance
# ═══════════════════════════════════════════════════════════════════

# Sina API column labels (Chinese) for each ranking type
_SINA_VOL_COLS = ["rank", "broker", "volume", "vol_change"]
_SINA_LONG_COLS = ["rank", "broker", "long_pos", "long_change"]
_SINA_SHORT_COLS = ["rank", "broker", "short_pos", "short_change"]


def fetch_symbol_position_data(
    contract: str,
    date: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch position ranking for a single contract from Sina Finance.

    Calls akshare.futures_hold_pos_sina() three times:
      1. "成交量"   → volume ranking
      2. "多单持仓" → long position ranking
      3. "空单持仓" → short position ranking

    Then merges into one unified DataFrame.

    Parameters
    ----------
    contract : str
        Futures contract code, e.g. "I2609", "RB2610", "J2609".
    date : str
        Date in "YYYYMMDD" format, e.g. "20260710".

    Returns
    -------
    pd.DataFrame or None
        Merged DataFrame with columns:
        [broker, volume, vol_change, long_pos, long_change, short_pos, short_change]
        Returns None if all three calls return empty data.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed. Run: pip install akshare")
        return None

    # Normalize contract case — Sina expects uppercase
    contract_upper = contract.upper()

    dfs = {}

    # Call 1: Volume ranking
    try:
        df_vol = ak.futures_hold_pos_sina(
            symbol="成交量", contract=contract_upper, date=date
        )
    except Exception as e:
        logger.warning(f"Volume fetch failed for {contract_upper}: {e}")
        df_vol = pd.DataFrame()

    time.sleep(AKSHARE_REQUEST_DELAY)

    # Call 2: Long position ranking
    try:
        df_long = ak.futures_hold_pos_sina(
            symbol="多单持仓", contract=contract_upper, date=date
        )
    except Exception as e:
        logger.warning(f"Long position fetch failed for {contract_upper}: {e}")
        df_long = pd.DataFrame()

    time.sleep(AKSHARE_REQUEST_DELAY)

    # Call 3: Short position ranking
    try:
        df_short = ak.futures_hold_pos_sina(
            symbol="空单持仓", contract=contract_upper, date=date
        )
    except Exception as e:
        logger.warning(
            f"Short position fetch failed for {contract_upper}: {e}")
        df_short = pd.DataFrame()

    # Check if we got any data
    all_empty = all(
        df is None or df.empty for df in [df_vol, df_long, df_short]
    )
    if all_empty:
        logger.warning(
            f"No data from Sina for {contract_upper} on {date}. "
            f"Possible reasons: non-trading day, contract expired, "
            f"or data not yet published."
        )
        return None

    # Rename and prepare each DataFrame
    def _prepare(df_raw, col_map, key_col="broker"):
        """Rename columns and set broker as index for merging."""
        if df_raw is None or df_raw.empty:
            return None
        # The Sina function returns columns with Chinese names.
        # The exact names may vary by akshare version, so we rename by position.
        df = df_raw.copy()
        actual_cols = list(df.columns)
        # Map the first N columns using the known schema
        rename = {}
        for i, new_name in enumerate(col_map):
            if i < len(actual_cols):
                rename[actual_cols[i]] = new_name
        df = df.rename(columns=rename)
        # Keep only our target columns (exclude rank column)
        keep = [c for c in col_map if c in df.columns and c != "rank"]
        df = df[keep].copy()
        # Convert numeric columns
        for col in keep:
            if col not in ("broker", "rank"):
                df[col] = pd.to_numeric(
                    df[col], errors="coerce").fillna(0).astype(int)
        # Remove aggregation/summary rows
        if "broker" in df.columns:
            summary_kw = ["合计", "总和", "总计"]
            df = df[~df["broker"].astype(str).str.strip().isin(summary_kw)]
        return df.set_index("broker")

    vol_df = _prepare(df_vol,   _SINA_VOL_COLS)
    long_df = _prepare(df_long,  _SINA_LONG_COLS)
    short_df = _prepare(df_short, _SINA_SHORT_COLS)

    # Merge: start with long positions, join short, then volume
    merged = None
    for part in [long_df, short_df, vol_df]:
        if part is None or part.empty:
            continue
        if merged is None:
            merged = part
        else:
            merged = merged.join(part, how="outer")

    if merged is None or merged.empty:
        return None

    # Fill missing values
    for col in ["volume", "vol_change", "long_pos", "long_change", "short_pos", "short_change"]:
        if col not in merged.columns:
            merged[col] = 0
    merged = merged.fillna(0).astype({c: int for c in merged.columns})

    # Reset index so broker is a column
    merged = merged.reset_index()
    merged = merged.rename(columns={"index": "broker"})

    # Sort by total position
    merged["_total"] = merged["long_pos"] + merged["short_pos"]
    merged = merged.sort_values(
        "_total", ascending=False).drop(columns=["_total"])
    merged = merged.reset_index(drop=True)

    logger.info(
        f"Fetched {contract_upper} on {date}: "
        f"{len(merged)} brokers, "
        f"long={merged['long_pos'].sum()}, short={merged['short_pos'].sum()}"
    )
    return merged


# ═══════════════════════════════════════════════════════════════════
# 2. Clean and store
# ═══════════════════════════════════════════════════════════════════

def clean_and_store(
    df: pd.DataFrame,
    symbol: str,
    date: str,
) -> Optional[dict]:
    """
    Store cleaned position data to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Already-merged DataFrame from fetch_symbol_position_data().
    symbol : str
        Symbol code (e.g., "i", "rb", "j").
    date : str
        Date string in "YYYYMMDD" format.

    Returns
    -------
    dict or None
        {"symbol": str, "date": str, "total_long": int, "total_short": int,
         "broker_count": int}
        None if data is empty.
    """
    if df is None or df.empty:
        logger.warning(f"clean_and_store: empty DataFrame for {symbol}")
        return None

    # Ensure required columns exist
    required = ["broker"]
    for col in ["volume", "vol_change", "long_pos", "long_change", "short_pos", "short_change"]:
        if col in df.columns:
            required.append(col)

    df_clean = df[required].copy()

    # Ensure numeric types
    for col in required:
        if col != "broker":
            df_clean[col] = pd.to_numeric(
                df_clean[col], errors="coerce").fillna(0).astype(int)

    # Save to Parquet
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, f"{symbol}_{date}.parquet")
    df_clean.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df_clean)} rows → {output_path}")

    total_long = int(df_clean["long_pos"].sum()
                     ) if "long_pos" in df_clean.columns else 0
    total_short = int(df_clean["short_pos"].sum()
                      ) if "short_pos" in df_clean.columns else 0

    return {
        "symbol": symbol,
        "date": date,
        "total_long": total_long,
        "total_short": total_short,
        "broker_count": len(df_clean),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Load stored data
# ═══════════════════════════════════════════════════════════════════

def load_stored_data(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """Load previously stored Parquet data for a symbol+date."""
    filepath = os.path.join(RAW_DATA_DIR, f"{symbol}_{date}.parquet")
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def load_historical_positions(
    symbol: str,
    lookback_days: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Load all available historical position data for a symbol.

    Scans data/raw/ for files matching `{symbol}_*.parquet`, loads each,
    and returns a dict mapping date string → DataFrame sorted by date ascending.

    Parameters
    ----------
    symbol : str
        Symbol code (e.g., "i", "rb", "j").
    lookback_days : int
        Max number of recent files to load (default 20).

    Returns
    -------
    dict[str, pd.DataFrame]
        {"20260701": DataFrame, "20260702": DataFrame, ...}
        Sorted by date ascending. Empty dict if no files found.
    """
    if not os.path.isdir(RAW_DATA_DIR):
        logger.warning(f"Raw data directory not found: {RAW_DATA_DIR}")
        return {}

    # Find all parquet files for this symbol
    prefix = f"{symbol}_"
    files = []
    for fname in os.listdir(RAW_DATA_DIR):
        if fname.startswith(prefix) and fname.endswith(".parquet"):
            # Extract date from filename: {symbol}_{YYYYMMDD}.parquet
            date_part = fname[len(prefix):-len(".parquet")]
            if len(date_part) == 8 and date_part.isdigit():
                files.append((date_part, os.path.join(RAW_DATA_DIR, fname)))

    if not files:
        logger.warning(f"No parquet files found for symbol '{symbol}'")
        return {}

    # Sort by date descending, take most recent N
    files.sort(key=lambda x: x[0], reverse=True)
    files = files[:lookback_days]
    # Re-sort ascending for time-series analysis
    files.sort(key=lambda x: x[0])

    result = {}
    for date_str, filepath in files:
        try:
            df = pd.read_parquet(filepath)
            if df is not None and not df.empty:
                result[date_str] = df
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    logger.info(
        f"Loaded {len(result)} historical files for '{symbol}' "
        f"(requested {lookback_days})"
    )
    return result


def list_available_dates(symbol: str) -> list[str]:
    """
    List all available dates for a symbol, sorted ascending.

    Parameters
    ----------
    symbol : str
        Symbol code.

    Returns
    -------
    list[str]
        Date strings in "YYYYMMDD" format, sorted ascending.
    """
    if not os.path.isdir(RAW_DATA_DIR):
        return []

    prefix = f"{symbol}_"
    dates = []
    for fname in os.listdir(RAW_DATA_DIR):
        if fname.startswith(prefix) and fname.endswith(".parquet"):
            date_part = fname[len(prefix):-len(".parquet")]
            if len(date_part) == 8 and date_part.isdigit():
                dates.append(date_part)

    dates.sort()
    return dates


def load_latest_bias() -> dict:
    """Load the daily_bias.json file. Returns empty dict if not found."""
    bias_path = os.path.join(BIAS_DATA_DIR, BIAS_FILE_NAME)
    if not os.path.exists(bias_path):
        logger.warning(f"Bias file not found: {bias_path}")
        return {}
    try:
        with open(bias_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load bias file: {e}")
        return {}


def get_latest_bias_for_symbol(symbol: str) -> dict:
    """Get the latest daily bias for a specific symbol."""
    all_bias = load_latest_bias()
    if not all_bias:
        return {}
    # Sort dates by actual datetime, not string
    valid_dates = []
    for d in all_bias:
        try:
            # Handle both "2026-07-10" and malformed "2026-710"
            parts = d.split("-")
            if len(parts) == 3:
                valid_dates.append((datetime.strptime(d, "%Y-%m-%d"), d))
            elif len(parts) == 2:
                # Malformed like "2026-710" → try to parse as YYYY-MMDD
                clean = f"{parts[0]}-{parts[1][:2]}-{parts[1][2:]}" if len(parts[1]) >= 4 else None
                if clean:
                    valid_dates.append((datetime.strptime(clean, "%Y-%m-%d"), d))
        except (ValueError, IndexError):
            continue
    if not valid_dates:
        return {}
    # Pick latest by actual date
    latest_key = max(valid_dates, key=lambda x: x[0])[1]
    return all_bias[latest_key]


# ═══════════════════════════════════════════════════════════════════
# 4. Main pipeline: fetch → clean → compute bias
# ═══════════════════════════════════════════════════════════════════

def fetch_and_process_all(date: Optional[str] = None) -> dict[str, Optional[dict]]:
    """
    Full pipeline: fetch position data per symbol → clean → compute daily bias.

    Parameters
    ----------
    date : str, optional
        "YYYYMMDD" format. Defaults to today.

    Returns
    -------
    dict[str, Optional[dict]]
        {symbol: bias_dict} for each symbol. None for symbols with no data.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    date_display = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(
        date) == 8 else date
    logger.info(f"=== Fetch pipeline for {date} ===")

    # ── Step 1: Fetch per-symbol via Sina ──
    summaries = {}
    for symbol, info in MONITORED_SYMBOLS.items():
        contract = info.get("contract", "")
        if not contract:
            logger.warning(f"No contract configured for {symbol}")
            summaries[symbol] = None
            continue

        time.sleep(AKSHARE_REQUEST_DELAY)
        df = fetch_symbol_position_data(contract, date)

        if df is None or df.empty:
            summaries[symbol] = None
            logger.warning(f"  {symbol} ({contract}): no data")
            continue

        summary = clean_and_store(df, symbol, date)
        summaries[symbol] = summary

    # ── Step 2: Compute daily bias ──
    today = datetime.strptime(date, "%Y%m%d")
    prev_date = _get_previous_trading_day(today)

    bias_results = {}
    for symbol in MONITORED_SYMBOLS:
        if summaries[symbol] is None:
            bias_results[symbol] = None
            continue

        current_df = load_stored_data(symbol, date)
        if current_df is None or current_df.empty:
            bias_results[symbol] = None
            continue

        prev_df = load_stored_data(symbol, prev_date) if prev_date else None

        cr = calc_concentration_ratio(current_df)
        delta_np = calc_net_position_momentum(current_df, prev_df)

        bias = calc_daily_bias(
            symbol=symbol,
            cr_long=cr["cr_long"],
            cr_short=cr["cr_short"],
            delta_np=delta_np,
            date=date_display,
        )
        bias["top_long_cr_brokers"] = cr["top_long_brokers"]
        bias["top_short_cr_brokers"] = cr["top_short_brokers"]

        bias_results[symbol] = bias
        logger.info(
            f"  {symbol} bias: {bias['bias'].upper()} "
            f"(conv={bias['conviction']:.2f})"
        )

    # ── Step 3: Save bias ──
    _save_bias(bias_results, date_display)
    return bias_results


def _get_previous_trading_day(today: datetime) -> Optional[str]:
    """Estimate previous trading day (skip weekends, not holidays)."""
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev.strftime("%Y%m%d")


def _save_bias(bias_results: dict, date: str) -> None:
    """Merge new bias data into daily_bias.json. Cleans malformed date keys."""
    os.makedirs(BIAS_DATA_DIR, exist_ok=True)
    bias_path = os.path.join(BIAS_DATA_DIR, BIAS_FILE_NAME)

    all_bias = load_latest_bias()

    # 清理异常日期键（如 "2026-710"）
    clean_bias = {}
    for k, v in all_bias.items():
        try:
            parts = k.split("-")
            if len(parts) == 3 and len(parts[1]) == 2:
                datetime.strptime(k, "%Y-%m-%d")
                clean_bias[k] = v
        except (ValueError, IndexError):
            logger.warning(f"Removing malformed date key: {k}")
    all_bias = clean_bias

    if date not in all_bias:
        all_bias[date] = {}

    for symbol, bias_dict in bias_results.items():
        if bias_dict is not None:
            all_bias[date][symbol] = bias_dict

    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(all_bias, f, ensure_ascii=False, indent=2)

    logger.info(f"Bias saved → {bias_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. Daily Price Data Fetching
# ═══════════════════════════════════════════════════════════════════

def fetch_daily_prices(
    symbol: str,
    start_date: str = "20260101",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily futures price data via AkShare (Sina Finance).

    Uses the continuous contract symbol (e.g., "I0" for iron ore,
    "RB0" for rebar) to get a long history without contract rollover gaps.

    Parameters
    ----------
    symbol : str
        Symbol code (e.g., "i", "rb"). The function maps this to
        the Sina continuous contract code.
    start_date : str
        Start date in "YYYYMMDD" format. Default "20260101".
    end_date : str, optional
        End date in "YYYYMMDD" format. Defaults to today.

    Returns
    -------
    pd.DataFrame or None
        Columns: [date, open, high, low, close, volume, hold]
        (hold = open interest / 持仓量)
        Returns None if fetch fails or data is empty.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed. Run: pip install akshare")
        return None

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # Map symbol to Sina continuous contract code
    symbol_upper = symbol.upper()
    # Common continuous codes: commodity letter + "0"
    continuous_map = {
        # 黑色系
        "I": "I0", "RB": "RB0", "J": "J0",
        "HC": "HC0", "JM": "JM0", "FG": "FG0",
        "SA": "SA0", "SM": "SM0", "SF": "SF0",
        # 油脂
        "Y": "Y0", "P": "P0", "OI": "OI0",
        # 农产品
        "M": "M0", "RM": "RM0", "A": "A0", "B": "B0",
        "C": "C0", "CS": "CS0", "JD": "JD0", "LH": "LH0",
        # 软商品
        "SR": "SR0", "CF": "CF0", "AP": "AP0",
        "RU": "RU0", "NR": "NR0", "SP": "SP0", "UR": "UR0",
        # 化工
        "MA": "MA0", "TA": "TA0", "EG": "EG0", "EB": "EB0",
        "PP": "PP0", "L": "L0", "V": "V0",
        # 能源
        "SC": "SC0", "FU": "FU0", "PG": "PG0",
    }
    sina_symbol = continuous_map.get(symbol_upper, f"{symbol_upper}0")

    try:
        df = ak.futures_zh_daily_sina(symbol=sina_symbol)
    except Exception as e:
        logger.warning(f"Failed to fetch daily prices for {sina_symbol}: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"No price data returned for {sina_symbol}")
        return None

    # Normalize columns — AkShare returns Chinese column names
    # Expected: ["date", "open", "high", "low", "close", "volume", "hold"]
    col_map = {
        "日期": "date", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close", "成交量": "volume",
        "持仓量": "hold",
    }
    df = df.rename(columns=col_map)

    # Ensure standard columns exist
    for std_col in ["date", "open", "high", "low", "close"]:
        if std_col not in df.columns:
            logger.warning(f"Missing column in price data: {std_col}")
            return None

    # Parse date and filter range
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    start_dt = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")

    if start_dt:
        df = df[df["date"] >= start_dt]
    if end_dt:
        df = df[df["date"] <= end_dt]

    if df.empty:
        return None

    # Sort by date, reset index
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric types
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["volume", "hold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    logger.info(
        f"Fetched {len(df)} daily bars for {sina_symbol} "
        f"({df['date'].min().date()} → {df['date'].max().date()})"
    )
    return df


def load_price_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load cached daily price data from parquet.

    Parameters
    ----------
    symbol : str
        Symbol code (e.g., "i", "rb").

    Returns
    -------
    pd.DataFrame or None
    """
    filepath = os.path.join(PRICE_DATA_DIR, f"{symbol}_daily.parquet")
    if os.path.exists(filepath):
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"Failed to load price data: {e}")
    return None


def fetch_and_store_prices(
    symbol: str,
    start_date: str = "20260101",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily prices and cache to parquet.

    Parameters
    ----------
    symbol : str
        Symbol code.
    start_date : str
        Start date "YYYYMMDD".
    end_date : str, optional
        End date "YYYYMMDD".

    Returns
    -------
    pd.DataFrame or None
    """
    df = fetch_daily_prices(symbol, start_date, end_date)
    if df is None or df.empty:
        return None

    os.makedirs(PRICE_DATA_DIR, exist_ok=True)
    filepath = os.path.join(PRICE_DATA_DIR, f"{symbol}_daily.parquet")
    df.to_parquet(filepath, index=False)
    logger.info(f"Price data cached → {filepath}")
    return df


def fetch_prices_for_all_monitored(
    start_date: str = "20260101",
) -> dict[str, Optional[pd.DataFrame]]:
    """
    Fetch and cache daily prices for all monitored symbols.

    Returns dict[symbol, DataFrame].
    """
    results = {}
    for symbol in MONITORED_SYMBOLS:
        time.sleep(AKSHARE_REQUEST_DELAY)
        df = fetch_and_store_prices(symbol, start_date=start_date)
        results[symbol] = df
        if df is not None:
            print(f"  {symbol}: {len(df)} bars")
        else:
            print(f"  {symbol}: FAILED")
    return results


# ═══════════════════════════════════════════════════════════════════
# 6. Batch Historical Data Fetching
# ═══════════════════════════════════════════════════════════════════

def fetch_batch_range(
    start_date: str,
    end_date: Optional[str] = None,
    symbols: Optional[list[str]] = None,
) -> dict[str, int]:
    """
    Batch-fetch position data over a date range, skipping weekends.

    For each calendar date between start_date and end_date:
      - Skip if Saturday (weekday 5) or Sunday (weekday 6)
      - Fetch position data for all (or specified) symbols
      - If ALL symbols return empty → likely a holiday, log and skip
      - Auto-save to parquet and update bias file

    Parameters
    ----------
    start_date : str
        Start date "YYYYMMDD".
    end_date : str, optional
        End date "YYYYMMDD". Defaults to today.
    symbols : list[str], optional
        Symbols to fetch. Defaults to all MONITORED_SYMBOLS.

    Returns
    -------
    dict[str, int]
        {date_str: successful_symbol_count}.
        Keys are dates where at least one symbol had data.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if symbols is None:
        symbols = list(MONITORED_SYMBOLS.keys())

    # Validate symbols
    invalid = [s for s in symbols if s not in MONITORED_SYMBOLS]
    if invalid:
        logger.warning(f"Ignoring unknown symbols: {invalid}")
        symbols = [s for s in symbols if s in MONITORED_SYMBOLS]
    if not symbols:
        logger.error("No valid symbols to fetch.")
        return {}

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    total_days = (end_dt - start_dt).days + 1
    skipped_weekends = 0
    skipped_holidays = 0
    processed_days = 0
    results = {}

    print(f"\n{'='*60}")
    print(f"Batch Fetch: {start_dt.date()} → {end_dt.date()}")
    print(f"Symbols: {len(symbols)} ({', '.join(symbols[:6])}"
          f"{'...' if len(symbols) > 6 else ''})")
    print(f"Total calendar days: {total_days}")
    print(f"{'='*60}\n")

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")

        # Skip weekends
        if current.weekday() >= 5:
            day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            print(f"  [{date_str}] {day_name[current.weekday()]} — 周末，跳过")
            skipped_weekends += 1
            current += timedelta(days=1)
            continue

        # Fetch for this date
        print(f"  [{date_str}] 抓取中...", end=" ", flush=True)

        from src.indicators import (
            calc_concentration_ratio,
            calc_daily_bias,
            calc_net_position_momentum,
        )

        success_count = 0
        date_summaries = {}

        for sym in symbols:
            info = MONITORED_SYMBOLS[sym]
            contract = info.get("contract", "")
            if not contract:
                continue

            try:
                df = fetch_symbol_position_data(contract, date_str)
            except Exception as e:
                logger.debug(f"Fetch error {sym}: {e}")
                df = None

            if df is None or df.empty:
                continue

            summary = clean_and_store(df, sym, date_str)
            if summary:
                date_summaries[sym] = summary
                success_count += 1

            time.sleep(AKSHARE_REQUEST_DELAY)

        # Check: did we get any data at all?
        if success_count == 0:
            print(f"0/{len(symbols)} — 无数据（可能节假日），跳过")
            skipped_holidays += 1
            current += timedelta(days=1)
            continue

        # Compute bias for this date
        prev_date = _get_previous_trading_day(current)
        bias_results = {}

        for sym in symbols:
            if sym not in date_summaries:
                bias_results[sym] = None
                continue

            current_df = load_stored_data(sym, date_str)
            prev_df = load_stored_data(sym, prev_date) if prev_date else None

            if current_df is None or current_df.empty:
                bias_results[sym] = None
                continue

            cr = calc_concentration_ratio(current_df)
            delta_np = calc_net_position_momentum(current_df, prev_df)

            date_display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            bias = calc_daily_bias(
                symbol=sym,
                cr_long=cr["cr_long"],
                cr_short=cr["cr_short"],
                delta_np=delta_np,
                date=date_display,
            )
            bias["top_long_cr_brokers"] = cr["top_long_brokers"]
            bias["top_short_cr_brokers"] = cr["top_short_brokers"]
            bias_results[sym] = bias

        _save_bias(bias_results, date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:])

        print(f"{success_count}/{len(symbols)} 成功")
        results[date_str] = success_count
        processed_days += 1

        current += timedelta(days=1)

    # ── Summary ──
    total_success = sum(results.values())
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"  日历天数:     {total_days}")
    print(f"  跳过周末:     {skipped_weekends}")
    print(f"  跳过节假日:   {skipped_holidays}")
    print(f"  成功抓取:     {processed_days} 个交易日")
    print(f"  总数据条数:   {total_success}")
    print(f"{'='*60}\n")

    return results


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fetch futures position ranking and compute daily bias."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date in YYYYMMDD format (default: today).",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYYMMDD for batch mode. Requires --end.",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYYMMDD for batch mode. Defaults to today.",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbol list (e.g. 'i,rb,j'). Default: all.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Batch mode ──
    if args.start:
        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        fetch_batch_range(
            start_date=args.start,
            end_date=args.end,
            symbols=symbols,
        )
        return

    # ── Single-date mode ──
    results = fetch_and_process_all(date=args.date)

    success = sum(1 for v in results.values() if v is not None)
    print(f"\nDone. {success}/{len(results)} symbols processed.")
    for symbol, bias in results.items():
        name = MONITORED_SYMBOLS[symbol]["name"]
        if bias:
            print(f"  {symbol} ({name}): {bias['bias'].upper()} "
                  f"(conv={bias['conviction']:.2f})")
        else:
            print(f"  {symbol} ({name}): NO DATA")


if __name__ == "__main__":
    main()
