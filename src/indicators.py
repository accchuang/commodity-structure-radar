"""
indicators.py — Core factor calculations for institutional bias & stop-run detection.

All functions are PURE: no I/O, no side effects, no global state.
Testable in isolation with synthetic data.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    CR_HIGH_THRESHOLD,
    CR_LOW_THRESHOLD,
    DEFAULT_WICK_BODY_RATIO,
    HVN_VOLUME_RATIO,
    POC_LOOKBACK_BARS,
    POC_PRICE_BUCKETS,
    STOPRUN_CONFIRM_WINDOW,
    TOP_N_CONCENTRATION,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Concentration Ratio
# ═══════════════════════════════════════════════════════════════════

def calc_concentration_ratio(
    df: pd.DataFrame,
    top_n: int = TOP_N_CONCENTRATION,
) -> dict:
    """
    Calculate long/short concentration ratio from cleaned ranking data.

    CR_long  = sum(top_n long positions)  / sum(all long positions)
    CR_short = sum(top_n short positions) / sum(all short positions)

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ['broker', 'long_pos', 'short_pos'].
        Rows are institutional brokers.
    top_n : int
        Number of top brokers to include. Defaults to TOP_N_CONCENTRATION (3).

    Returns
    -------
    dict
        {
            "cr_long": float,            # 0.0–1.0
            "cr_short": float,           # 0.0–1.0
            "top_long_brokers": [str],   # broker names with largest long_pos
            "top_short_brokers": [str],  # broker names with largest short_pos
        }
    """
    if df is None or df.empty:
        logger.warning("calc_concentration_ratio: empty DataFrame, returning zeros")
        return {
            "cr_long": 0.0,
            "cr_short": 0.0,
            "top_long_brokers": [],
            "top_short_brokers": [],
        }

    total_long = df["long_pos"].sum()
    total_short = df["short_pos"].sum()

    # Top-N by long position
    top_long_df = df.nlargest(min(top_n, len(df)), "long_pos")
    top_long_sum = top_long_df["long_pos"].sum()
    top_long_brokers = top_long_df["broker"].tolist()

    # Top-N by short position
    top_short_df = df.nlargest(min(top_n, len(df)), "short_pos")
    top_short_sum = top_short_df["short_pos"].sum()
    top_short_brokers = top_short_df["broker"].tolist()

    cr_long = top_long_sum / total_long if total_long > 0 else 0.0
    cr_short = top_short_sum / total_short if total_short > 0 else 0.0

    return {
        "cr_long": round(cr_long, 4),
        "cr_short": round(cr_short, 4),
        "top_long_brokers": top_long_brokers,
        "top_short_brokers": top_short_brokers,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. Net Position Momentum (ΔNP)
# ═══════════════════════════════════════════════════════════════════

def calc_net_position_momentum(
    current_df: pd.DataFrame,
    prev_df: Optional[pd.DataFrame] = None,
) -> dict[str, int]:
    """
    Calculate ΔNP (net position momentum) for each institution.

    ΔNP = (long_cur − short_cur) − (long_prev − short_prev)

    Positive ΔNP → institution is adding net longs / covering shorts.
    Negative ΔNP → institution is adding net shorts / liquidating longs.

    Parameters
    ----------
    current_df : pd.DataFrame
        Today's cleaned data with columns: ['broker', 'long_pos', 'short_pos'].
    prev_df : pd.DataFrame or None
        Yesterday's cleaned data. None if no previous data available.

    Returns
    -------
    dict[str, int]
        {broker_name: delta_net_position}, sorted by |ΔNP| descending.
        Empty dict if current_df is empty.
    """
    if current_df is None or current_df.empty:
        return {}

    # Compute today's net position per broker
    current_net = current_df.set_index("broker")
    current_net["net_cur"] = current_net["long_pos"] - current_net["short_pos"]

    if prev_df is not None and not prev_df.empty:
        prev_net = prev_df.set_index("broker")
        prev_net["net_prev"] = prev_net["long_pos"] - prev_net["short_pos"]

        # Outer join — brokers may enter or leave the report
        merged = pd.merge(
            current_net[["net_cur"]],
            prev_net[["net_prev"]],
            left_index=True,
            right_index=True,
            how="outer",
        ).fillna(0)

        merged["delta_np"] = (merged["net_cur"] - merged["net_prev"]).astype(int)
    else:
        # No previous data: ΔNP is simply today's net position
        current_net["delta_np"] = current_net["net_cur"].astype(int)
        merged = current_net

    # Sort by absolute delta (largest moves first)
    result = (
        merged["delta_np"]
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .to_dict()
    )

    return result


# ═══════════════════════════════════════════════════════════════════
# 3. Key Level Extraction
# ═══════════════════════════════════════════════════════════════════

def calc_key_levels(df_1h: pd.DataFrame) -> dict:
    """
    Extract key price levels from hourly K-line data.

    Levels extracted:
    - prev_high / prev_low: previous complete trading day's high/low
    - poc: Volume Profile Point of Control (price with highest volume)
    - hvn_upper / hvn_lower: High Volume Node bounds
    - today_open: current session's opening price

    Parameters
    ----------
    df_1h : pd.DataFrame
        Hourly K-line data with columns:
        ['open', 'high', 'low', 'close', 'volume', 'datetime'].
        Must span at least one complete trading day.

    Returns
    -------
    dict
        {
            "prev_high": float,
            "prev_low": float,
            "poc": float,
            "hvn_upper": float,
            "hvn_lower": float,
            "today_open": float,
        }
        Empty dict if df_1h is empty or has insufficient data.
    """
    if df_1h is None or df_1h.empty:
        return {}

    # Ensure datetime column
    if "datetime" not in df_1h.columns:
        return {}

    df = df_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ── Previous Day High / Low ──
    today = df["datetime"].max().date()
    df["date"] = df["datetime"].dt.date
    prev_day_df = df[df["date"] < today]

    prev_high = None
    prev_low = None
    if not prev_day_df.empty:
        prev_high = float(prev_day_df["high"].max())
        prev_low = float(prev_day_df["low"].min())

    # ── Today's Open ──
    today_df = df[df["date"] == today]
    today_open = None
    if not today_df.empty:
        today_open = float(today_df.iloc[0]["open"])

    # ── Volume Profile POC (last N bars) ──
    recent = df.tail(POC_LOOKBACK_BARS)
    if recent.empty or "volume" not in recent.columns:
        return _build_levels_dict(prev_high, prev_low, None, None, None, today_open)

    price_min = recent["low"].min()
    price_max = recent["high"].max()

    if price_max <= price_min:
        return _build_levels_dict(prev_high, prev_low, None, None, None, today_open)

    bucket_size = (price_max - price_min) / POC_PRICE_BUCKETS
    if bucket_size <= 0:
        return _build_levels_dict(prev_high, prev_low, None, None, None, today_open)

    buckets = np.arange(price_min, price_max + bucket_size, bucket_size)
    volume_profile = np.zeros(len(buckets) - 1)

    for _, row in recent.iterrows():
        candle_low = row["low"]
        candle_high = row["high"]
        candle_vol = row["volume"]

        # Distribute volume evenly across the candle's range
        candle_buckets = max(1, int((candle_high - candle_low) / bucket_size))
        vol_per_bucket = candle_vol / candle_buckets

        for cb in range(candle_buckets):
            price = candle_low + cb * bucket_size
            idx = min(int((price - price_min) / bucket_size), len(volume_profile) - 1)
            if 0 <= idx < len(volume_profile):
                volume_profile[idx] += vol_per_bucket

    # POC = bucket center with max volume
    max_vol_idx = int(np.argmax(volume_profile))
    poc = float(buckets[max_vol_idx] + bucket_size / 2)

    # HVN = contiguous region where volume >= 70% of max
    max_vol = volume_profile[max_vol_idx]
    hvn_threshold = max_vol * HVN_VOLUME_RATIO
    hvn_mask = volume_profile >= hvn_threshold

    # Find contiguous HVN region around POC
    hvn_lower = poc
    hvn_upper = poc
    left = max_vol_idx
    right = max_vol_idx

    while left > 0 and hvn_mask[left - 1]:
        left -= 1
    while right < len(hvn_mask) - 1 and hvn_mask[right + 1]:
        right += 1

    hvn_lower = float(buckets[left])
    hvn_upper = float(min(buckets[right] + bucket_size, price_max))

    return _build_levels_dict(
        prev_high, prev_low, round(poc, 1),
        round(hvn_lower, 1), round(hvn_upper, 1),
        today_open,
    )


def _build_levels_dict(prev_high, prev_low, poc, hvn_lower, hvn_upper, today_open):
    """Helper: build the levels dict, omitting None values."""
    result = {}
    if prev_high is not None:
        result["prev_high"] = prev_high
    if prev_low is not None:
        result["prev_low"] = prev_low
    if poc is not None:
        result["poc"] = poc
    if hvn_lower is not None:
        result["hvn_lower"] = hvn_lower
    if hvn_upper is not None:
        result["hvn_upper"] = hvn_upper
    if today_open is not None:
        result["today_open"] = today_open
    return result


# ═══════════════════════════════════════════════════════════════════
# 4. Stop Run Detection — THE CORE SIGNAL
# ═══════════════════════════════════════════════════════════════════

def detect_stop_run(
    candles: pd.DataFrame,
    level: float,
    direction: str,
    symbol: str = "",
    wick_ratio: float = DEFAULT_WICK_BODY_RATIO,
    max_candles: int = STOPRUN_CONFIRM_WINDOW,
) -> tuple[bool, Optional[dict]]:
    """
    Detect a Stop Run (false breakout / liquidity sweep) at a key level.

    This is the core anti-intuitive signal engine.
    It does NOT use bid/ask depth — pure price action at key levels.

    State Machine:
        IDLE → PIERCE (price crosses level)
             → MONITOR (within max_candles bars)
             → CONFIRMED (close rejects back + wick threshold met)
             | EXPIRED (window exceeded, no reversal)

    Parameters
    ----------
    candles : pd.DataFrame
        Sequential candles ordered by time ascending.
        Must have: ['open', 'high', 'low', 'close', 'datetime'].
    level : float
        The key level being tested.
    direction : str
        "long"  (bullish)  → price pierces BELOW support level, then rejects upward.
        "short" (bearish)  → price pierces ABOVE resistance level, then rejects downward.
    symbol : str
        Symbol label for alert metadata.
    wick_ratio : float
        Minimum wick/(high-low) ratio for a valid rejection candle.
    max_candles : int
        Max candles after the first pierce to confirm rejection.

    Returns
    -------
    (is_stop_run, alert_details)
        is_stop_run : bool
        alert_details : dict or None
            If confirmed:
            {
                "symbol": str,
                "direction": "long" | "short",
                "level": float,
                "level_name": str,
                "pierce_time": datetime,
                "reject_time": datetime,
                "pierce_price": float,
                "close_price": float,
                "wick_ratio_observed": float,
                "candle_count": int,
                "severity": "high" | "medium",
            }
    """
    # ── Guard clauses ──
    if candles is None or candles.empty:
        return (False, None)
    if level is None or np.isnan(level):
        return (False, None)
    if direction not in ("long", "short"):
        logger.warning(f"detect_stop_run: invalid direction '{direction}'")
        return (False, None)

    required_cols = {"open", "high", "low", "close", "datetime"}
    if not required_cols.issubset(candles.columns):
        missing = required_cols - set(candles.columns)
        logger.warning(f"detect_stop_run: missing columns {missing}")
        return (False, None)

    if len(candles) < 2:
        return (False, None)

    # Ensure sorted by time
    df = candles.sort_values("datetime").reset_index(drop=True)

    # ── Scan for pierce ──
    pierce_idx = -1
    pierce_price = 0.0

    if direction == "long":
        # Bullish stop-run: looking for price piercing BELOW support
        for i, row in df.iterrows():
            if row["low"] < level:
                pierce_idx = i
                pierce_price = row["low"]  # furthest extent below level
                break
    else:  # direction == "short"
        # Bearish stop-run: looking for price piercing ABOVE resistance
        for i, row in df.iterrows():
            if row["high"] > level:
                pierce_idx = i
                pierce_price = row["high"]  # furthest extent above level
                break

    if pierce_idx < 0:
        return (False, None)  # No pierce found

    pierce_candle = df.iloc[pierce_idx]
    pierce_time = pierce_candle["datetime"]

    # ── Monitor subsequent candles for rejection ──
    confirm_start = pierce_idx + 1
    confirm_end = min(pierce_idx + 1 + max_candles, len(df))

    if confirm_start >= confirm_end:
        return (False, None)  # No candles after pierce

    for ci in range(confirm_start, confirm_end):
        candle = df.iloc[ci]
        candle_range = candle["high"] - candle["low"]

        if candle_range <= 0:
            continue  # skip degenerate candles

        if direction == "long":
            # Bullish stop-run: price pierced BELOW support, must reject back ABOVE.
            # Rejection candle: close > level, and strong upward recovery from low.
            close_above_level = candle["close"] > level
            # Rejection ratio = (close - low) / range — captures full upward
            # recovery (both body and lower wick) from the probe low.
            rejection_ratio = (candle["close"] - candle["low"]) / candle_range

            if close_above_level and rejection_ratio >= wick_ratio:
                severity = "high" if rejection_ratio >= 0.75 else "medium"
                return (True, {
                    "symbol": symbol,
                    "direction": "long",
                    "level": level,
                    "level_name": "",  # filled by caller
                    "pierce_time": pierce_time,
                    "reject_time": candle["datetime"],
                    "pierce_price": pierce_price,
                    "close_price": candle["close"],
                    "wick_ratio_observed": round(rejection_ratio, 3),
                    "candle_count": ci - pierce_idx,
                    "severity": severity,
                })

        else:  # direction == "short"
            # Bearish stop-run: price pierced ABOVE resistance, must reject back BELOW.
            # Rejection candle: close < level, and strong downward rejection from high.
            close_below_level = candle["close"] < level
            # Rejection ratio = (high - close) / range — captures full downward
            # rejection (both body and upper wick) from the probe high.
            rejection_ratio = (candle["high"] - candle["close"]) / candle_range

            if close_below_level and rejection_ratio >= wick_ratio:
                severity = "high" if rejection_ratio >= 0.75 else "medium"
                return (True, {
                    "symbol": symbol,
                    "direction": "short",
                    "level": level,
                    "level_name": "",  # filled by caller
                    "pierce_time": pierce_time,
                    "reject_time": candle["datetime"],
                    "pierce_price": pierce_price,
                    "close_price": candle["close"],
                    "wick_ratio_observed": round(rejection_ratio, 3),
                    "candle_count": ci - pierce_idx,
                    "severity": severity,
                })

    # Window expired, no valid rejection
    return (False, None)


# ═══════════════════════════════════════════════════════════════════
# 5. Daily Bias Synthesis
# ═══════════════════════════════════════════════════════════════════

def calc_daily_bias(
    symbol: str,
    cr_long: float,
    cr_short: float,
    delta_np: dict[str, int],
    date: str,
) -> dict:
    """
    Synthesize institutional metrics into an actionable daily bias.

    Synthesis Rules (priority order):
    ┌──────────────────────────────────────────────────────────┐
    │ CR Condition                    │ ΔNP           │ Bias   │
    ├──────────────────────────────────────────────────────────┤
    │ CR_long >  0.4 AND > CR_short   │ aggregate > 0 │ BULLISH│
    │ CR_short > 0.4 AND > CR_long    │ aggregate < 0 │ BEARISH│
    │ CR_long > 0.4 AND CR_short > 0.4│ any           │ NEUTRAL│
    │ Both CR < 0.2                   │ any           │ SKIP   │
    │ Everything else                 │ any           │ NEUTRAL│
    └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    symbol : str
        Futures symbol code (e.g., "i", "rb", "j").
    cr_long : float
        Long concentration ratio from calc_concentration_ratio().
    cr_short : float
        Short concentration ratio from calc_concentration_ratio().
    delta_np : dict[str, int]
        {broker: delta_np} from calc_net_position_momentum().
    date : str
        "YYYY-MM-DD" format.

    Returns
    -------
    dict
        {
            "symbol": str,
            "date": str,
            "bias": "bullish" | "bearish" | "neutral" | "skip",
            "conviction": float (0.0–1.0),
            "cr_long": float,
            "cr_short": float,
            "aggregate_delta_np": int,
            "top_bullish": [str],
            "top_bearish": [str],
            "details": str (human-readable rationale),
        }
    """
    # ── Determine bias direction ──
    aggregate_dnp = sum(delta_np.values()) if delta_np else 0

    if cr_long > CR_HIGH_THRESHOLD and cr_long > cr_short and aggregate_dnp > 0:
        bias = "bullish"
        details = (
            f"多头集中度 {cr_long:.0%}，超过阈值 {CR_HIGH_THRESHOLD:.0%}，"
            f"且高于空头集中度 {cr_short:.0%}。"
            f"净持仓动量 +{aggregate_dnp}，主力持续加多。"
        )

    elif cr_short > CR_HIGH_THRESHOLD and cr_short > cr_long and aggregate_dnp < 0:
        bias = "bearish"
        details = (
            f"空头集中度 {cr_short:.0%}，超过阈值 {CR_HIGH_THRESHOLD:.0%}，"
            f"且高于多头集中度 {cr_long:.0%}。"
            f"净持仓动量 {aggregate_dnp}，主力持续加空。"
        )

    elif cr_long < CR_LOW_THRESHOLD and cr_short < CR_LOW_THRESHOLD:
        bias = "skip"
        details = (
            f"多空集中度均低于 {CR_LOW_THRESHOLD:.0%}（多头 {cr_long:.0%}，空头 {cr_short:.0%}）。"
            f"席位高度分散，无机构合力，今日不交易。"
        )

    elif cr_long > CR_HIGH_THRESHOLD and cr_short > CR_HIGH_THRESHOLD:
        bias = "neutral"
        details = (
            f"多空集中度双高（多头 {cr_long:.0%}，空头 {cr_short:.0%}），"
            f"均超过 {CR_HIGH_THRESHOLD:.0%}。"
            f"主力对决激烈，方向不明，谨慎观望。"
        )

    else:
        bias = "neutral"
        details = (
            f"信号混杂。多头集中度 {cr_long:.0%}，空头集中度 {cr_short:.0%}，"
            f"净持仓动量 {aggregate_dnp:+}。无明显方向性优势。"
        )

    # ── Compute conviction ──
    cr_diff = abs(cr_long - cr_short)
    max_abs_dnp = max(abs(v) for v in delta_np.values()) if delta_np else 1
    normalized_dnp = min(abs(aggregate_dnp) / max(max_abs_dnp, 1), 1.0)

    conviction = cr_diff * 0.7 + normalized_dnp * 0.3

    if bias in ("neutral", "skip"):
        conviction *= 0.5  # halve conviction for non-directional biases

    conviction = max(0.0, min(1.0, conviction))

    # ── Top bullish/bearish brokers ──
    sorted_by_dnp = sorted(delta_np.items(), key=lambda x: x[1], reverse=True)
    top_bullish = [broker for broker, dnp in sorted_by_dnp if dnp > 0][:3]
    top_bearish = [broker for broker, dnp in sorted_by_dnp if dnp < 0][:3]
    top_bearish.reverse()  # most negative first

    return {
        "symbol": symbol,
        "date": date,
        "bias": bias,
        "conviction": round(conviction, 3),
        "cr_long": cr_long,
        "cr_short": cr_short,
        "aggregate_delta_np": aggregate_dnp,
        "top_bullish": top_bullish,
        "top_bearish": top_bearish,
        "details": details,
    }
