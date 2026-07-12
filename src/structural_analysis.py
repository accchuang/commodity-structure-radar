"""
structural_analysis.py — Historical position structure evolution analysis.

Answers three questions:
  1. Is the position structure tightening or loosening? (CR trend)
  2. Which brokers are shifting stance? (Broker-level momentum)
  3. How does position structure relate to price? (Seat-price correlation)

All functions are PURE: no I/O, no side effects. Data loading is done
by fetcher_akshare.load_historical_positions().

Typical pipeline:
    from src.fetcher_akshare import load_historical_positions
    from src.structural_analysis import (
        calc_cr_timeseries,
        calc_cr_trend,
        detect_structure_change,
        calc_broker_momentum_timeseries,
    )

    hist = load_historical_positions("rb", lookback_days=20)
    cr_df = calc_cr_timeseries(hist)
    cr_df = calc_cr_trend(cr_df)
    events = detect_structure_change(cr_df)
    broker_ts = calc_broker_momentum_timeseries(hist, top_n=5)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    BROKER_RANK_LOOKBACK,
    CR_HIGH_THRESHOLD,
    CR_LOOSENING_THRESHOLD,
    CR_LOW_THRESHOLD,
    CR_TIGHTENING_THRESHOLD,
    CR_TREND_WINDOW,
    HHI_HIGH_CONCENTRATION,
    TOP_N_CONCENTRATION,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Herfindahl-Hirschman Index (HHI)
# ═══════════════════════════════════════════════════════════════════

def calc_herfindahl(df: pd.DataFrame, side: str = "long") -> float:
    """
    Calculate Herfindahl-Hirschman Index for long or short positions.

    HHI = Σ(share_i²)  where share_i = broker_i_position / total_position

    HHI ranges from ~0 (fully dispersed) to 1.0 (single monopoly).
    More granular than CR — captures the entire distribution, not just top-N.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'broker' and either 'long_pos' or 'short_pos' columns.
    side : str
        "long" or "short".

    Returns
    -------
    float
        0.0–1.0. Returns 0.0 if data is empty or total position is zero.
    """
    col = "long_pos" if side == "long" else "short_pos"

    if df is None or df.empty or col not in df.columns:
        return 0.0

    total = df[col].sum()
    if total <= 0:
        return 0.0

    shares = df[col] / total
    hhi = float((shares ** 2).sum())
    return round(hhi, 6)


# ═══════════════════════════════════════════════════════════════════
# 2. CR Time Series
# ═══════════════════════════════════════════════════════════════════

def calc_cr_timeseries(
    historical_data: dict[str, pd.DataFrame],
    top_n: int = TOP_N_CONCENTRATION,
) -> pd.DataFrame:
    """
    Compute daily concentration metrics from historical position data.

    For each date, computes:
      - CR_long, CR_short (top-N concentration ratio)
      - HHI_long, HHI_short (Herfindahl index)
      - total_long, total_short (aggregate positions)
      - broker_count (number of reporting institutions)
      - top_long_brokers, top_short_brokers (broker names)

    Parameters
    ----------
    historical_data : dict[str, pd.DataFrame]
        {date_str: DataFrame} from load_historical_positions().
    top_n : int
        Number of top brokers for CR calculation.

    Returns
    -------
    pd.DataFrame
        Columns: date, cr_long, cr_short, cr_diff, hhi_long, hhi_short,
                 total_long, total_short, broker_count,
                 top_long_brokers, top_short_brokers.
        Sorted by date ascending. Empty DataFrame if no data.
    """
    if not historical_data:
        logger.warning("calc_cr_timeseries: empty historical_data")
        return pd.DataFrame()

    from src.indicators import calc_concentration_ratio

    rows = []
    for date_str, df in sorted(historical_data.items()):
        if df is None or df.empty:
            continue

        cr = calc_concentration_ratio(df, top_n=top_n)
        hhi_long = calc_herfindahl(df, "long")
        hhi_short = calc_herfindahl(df, "short")

        total_long = int(df["long_pos"].sum()) if "long_pos" in df.columns else 0
        total_short = int(df["short_pos"].sum()) if "short_pos" in df.columns else 0

        rows.append({
            "date": date_str,
            "cr_long": cr["cr_long"],
            "cr_short": cr["cr_short"],
            "cr_diff": round(cr["cr_long"] - cr["cr_short"], 4),
            "hhi_long": hhi_long,
            "hhi_short": hhi_short,
            "total_long": total_long,
            "total_short": total_short,
            "broker_count": len(df),
            "top_long_brokers": cr["top_long_brokers"],
            "top_short_brokers": cr["top_short_brokers"],
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["date"] = pd.to_datetime(
        df_out["date"], format="%Y%m%d", errors="coerce"
    )
    df_out = df_out.sort_values("date").reset_index(drop=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════
# 3. CR Trend — Slope, Acceleration, Structure Signal
# ═══════════════════════════════════════════════════════════════════

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling linear regression slope on a series.

    slope = Cov(x, y) / Var(x) where x = [0, 1, ..., window-1].

    Returns NaN for rows where the window is incomplete.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    if denom == 0:
        return pd.Series([np.nan] * len(series), index=series.index)

    # Rolling covariance / variance
    def _slope(window_vals):
        if len(window_vals) < window or np.any(np.isnan(window_vals)):
            return np.nan
        y_mean = window_vals.mean()
        cov = ((x - x_mean) * (window_vals - y_mean)).sum()
        return cov / denom

    return series.rolling(window=window, min_periods=window).apply(
        _slope, raw=True
    )


def calc_cr_trend(
    cr_df: pd.DataFrame,
    window: int = CR_TREND_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling trend metrics for CR time series.

    Adds columns:
      - cr_long_slope: per-day change rate of CR_long over rolling window
      - cr_short_slope: per-day change rate of CR_short over rolling window
      - cr_diff_slope: slope of (CR_long - CR_short)
      - cr_long_accel: change in cr_long_slope (2nd derivative)
      - cr_short_accel: change in cr_short_slope
      - total_long_pct_change: % change in total long positions over window
      - total_short_pct_change: % change in total short positions over window
      - structure_long: "tightening" | "loosening" | "stable" for long side
      - structure_short: "tightening" | "loosening" | "stable" for short side

    Parameters
    ----------
    cr_df : pd.DataFrame
        Output of calc_cr_timeseries(). Must have columns:
        cr_long, cr_short, total_long, total_short.
    window : int
        Rolling window size for trend calculation.

    Returns
    -------
    pd.DataFrame
        A COPY of cr_df with additional trend columns.
    """
    if cr_df is None or cr_df.empty:
        logger.warning("calc_cr_trend: empty input")
        return cr_df if cr_df is not None else pd.DataFrame()

    df = cr_df.copy()

    # ── Rolling slopes ──
    df["cr_long_slope"] = _rolling_slope(df["cr_long"], window)
    df["cr_short_slope"] = _rolling_slope(df["cr_short"], window)
    df["cr_diff_slope"] = _rolling_slope(df["cr_diff"], window)

    # ── Acceleration (2nd derivative = slope of slope) ──
    # Use a shorter window for acceleration to detect inflections
    accel_window = max(3, window // 3)
    df["cr_long_accel"] = _rolling_slope(
        df["cr_long_slope"].fillna(0), accel_window
    )
    df["cr_short_accel"] = _rolling_slope(
        df["cr_short_slope"].fillna(0), accel_window
    )

    # ── Position change over window ──
    df["total_long_pct_change"] = (
        df["total_long"].pct_change(periods=window - 1).fillna(0)
    )
    df["total_short_pct_change"] = (
        df["total_short"].pct_change(periods=window - 1).fillna(0)
    )

    # ── Structure signal classification ──
    def _classify(slope_series: pd.Series) -> pd.Series:
        """Classify each row as tightening / loosening / stable."""
        result = pd.Series("stable", index=slope_series.index)
        result[slope_series > CR_TIGHTENING_THRESHOLD] = "tightening"
        result[slope_series < CR_LOOSENING_THRESHOLD] = "loosening"
        # NaN → "stable" (insufficient data)
        result[slope_series.isna()] = "stable"
        return result

    df["structure_long"] = _classify(df["cr_long_slope"])
    df["structure_short"] = _classify(df["cr_short_slope"])

    return df


# ═══════════════════════════════════════════════════════════════════
# 4. Structure Change Detection — Regime Shift Events
# ═══════════════════════════════════════════════════════════════════

def detect_structure_change(
    cr_trend_df: pd.DataFrame,
) -> list[dict]:
    """
    Detect meaningful regime change events in position structure.

    Event types:
      - "long_loosening": CR_long was tightening/stable → now loosening
      - "short_loosening": CR_short was tightening/stable → now loosening
      - "long_tightening": CR_long was loosening/stable → now tightening
      - "short_tightening": CR_short was loosening/stable → now tightening
      - "cr_cross": CR_long crossed above/below CR_short (dominance flip)
      - "hhi_divergence": HHI trends diverge from CR (hidden concentration shift)
      - "position_surge": total position changes >20% rapidly

    Parameters
    ----------
    cr_trend_df : pd.DataFrame
        Output of calc_cr_trend(). Must have structure_long, structure_short,
        cr_diff, hhi_long, hhi_short, total_long, total_short columns.

    Returns
    -------
    list[dict]
        Each event: {
            "date": str,
            "type": str (event type),
            "side": "long" | "short" | "both",
            "severity": "high" | "medium" | "low",
            "detail": str (human-readable description),
            "metrics": dict (relevant numbers),
        }
    """
    if cr_trend_df is None or cr_trend_df.empty or len(cr_trend_df) < 3:
        return []

    df = cr_trend_df.copy()
    events = []

    # ── 1. Structure regime transitions ──
    for side in ["long", "short"]:
        col = f"structure_{side}"
        prev_col = df[col].shift(1)

        # Tightening → Loosening (most important signal)
        flip_to_loose = (prev_col == "tightening") & (df[col] == "loosening")
        for idx in df.index[flip_to_loose]:
            row = df.loc[idx]
            cr_val = row[f"cr_{side}"]
            slope = row[f"cr_{side}_slope"]
            events.append({
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                "type": f"{side}_loosening",
                "side": side,
                "severity": "high",
                "detail": (
                    f"{'多头' if side == 'long' else '空头'}持仓结构从收紧转为松动。"
                    f"CR从高位开始回落（当前{cr_val:.1%}，"
                    f"日变化率{slope:.3f}），集中度正在瓦解。"
                ),
                "metrics": {
                    f"cr_{side}": cr_val,
                    f"cr_{side}_slope": slope,
                    f"hhi_{side}": row.get(f"hhi_{side}"),
                },
            })

        # Stable → Loosening
        flip_stable_loose = (prev_col == "stable") & (df[col] == "loosening")
        for idx in df.index[flip_stable_loose]:
            row = df.loc[idx]
            events.append({
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                "type": f"{side}_loosening",
                "side": side,
                "severity": "medium",
                "detail": (
                    f"{'多头' if side == 'long' else '空头'}持仓开始出现松动迹象，"
                    f"集中度趋势转为下降。"
                ),
                "metrics": {
                    f"cr_{side}": row[f"cr_{side}"],
                    f"cr_{side}_slope": row[f"cr_{side}_slope"],
                },
            })

        # Loosening → Tightening (re-concentration)
        flip_to_tight = (prev_col == "loosening") & (df[col] == "tightening")
        for idx in df.index[flip_to_tight]:
            row = df.loc[idx]
            events.append({
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                "type": f"{side}_tightening",
                "side": side,
                "severity": "medium",
                "detail": (
                    f"{'多头' if side == 'long' else '空头'}持仓重新集中，"
                    f"主力可能正在建仓。"
                ),
                "metrics": {
                    f"cr_{side}": row[f"cr_{side}"],
                    f"cr_{side}_slope": row[f"cr_{side}_slope"],
                },
            })

    # ── 2. CR dominance cross ──
    df["cr_diff_prev"] = df["cr_diff"].shift(1)
    cross_over = (df["cr_diff_prev"] < 0) & (df["cr_diff"] > 0)
    cross_under = (df["cr_diff_prev"] > 0) & (df["cr_diff"] < 0)

    for idx in df.index[cross_over]:
        row = df.loc[idx]
        events.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "type": "cr_cross",
            "side": "long",
            "severity": "high",
            "detail": (
                f"多头集中度上穿空头集中度（CR_long={row['cr_long']:.1%} > "
                f"CR_short={row['cr_short']:.1%}），多方力量可能转为主导。"
            ),
            "metrics": {
                "cr_long": row["cr_long"],
                "cr_short": row["cr_short"],
                "cr_diff": row["cr_diff"],
            },
        })

    for idx in df.index[cross_under]:
        row = df.loc[idx]
        events.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "type": "cr_cross",
            "side": "short",
            "severity": "high",
            "detail": (
                f"空头集中度上穿多头集中度（CR_short={row['cr_short']:.1%} > "
                f"CR_long={row['cr_long']:.1%}），空方力量可能转为主导。"
            ),
            "metrics": {
                "cr_long": row["cr_long"],
                "cr_short": row["cr_short"],
                "cr_diff": row["cr_diff"],
            },
        })

    # ── 3. HHI-CR divergence (hidden shift) ──
    if "hhi_long" in df.columns and len(df) >= 5:
        hhi_slope = _rolling_slope(df["hhi_long"], min(5, len(df)))
        cr_slope = df["cr_long_slope"]
        # Divergence: HHI moving opposite to CR
        diverge = (hhi_slope * cr_slope) < 0
        # Only flag when both slopes are meaningful
        diverge = diverge & (abs(hhi_slope) > 0.001) & (abs(cr_slope) > 0.005)

        for idx in df.index[diverge]:
            row = df.loc[idx]
            hhi_s = hhi_slope[idx]
            cr_s = cr_slope[idx]
            events.append({
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                "type": "hhi_divergence",
                "side": "long",
                "severity": "medium",
                "detail": (
                    f"多头HHI与CR出现背离：HHI{'上升' if hhi_s > 0 else '下降'} "
                    f"但CR{'上升' if cr_s > 0 else '下降'}。"
                    f"Top-3之外的中小席位正在发生结构性变化。"
                ),
                "metrics": {
                    "hhi_long": row["hhi_long"],
                    "cr_long": row["cr_long"],
                    "hhi_slope": hhi_s,
                    "cr_slope": cr_s,
                },
            })

    # ── 4. Position surge / collapse ──
    if "total_long_pct_change" in df.columns:
        surge_threshold = 0.20
        for side in ["long", "short"]:
            col = f"total_{side}_pct_change"
            surge = df[col].abs() > surge_threshold
            for idx in df.index[surge]:
                row = df.loc[idx]
                pct = row[col]
                events.append({
                    "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                    "type": "position_surge",
                    "side": side,
                    "severity": "high" if abs(pct) > 0.35 else "medium",
                    "detail": (
                        f"{'多头' if side == 'long' else '空头'}总持仓{'激增' if pct > 0 else '骤降'} "
                        f"{abs(pct):.1%}，可能有大资金进出。"
                    ),
                    "metrics": {
                        f"total_{side}": row[f"total_{side}"],
                        "pct_change": pct,
                    },
                })

    # Sort by date, then severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    events.sort(key=lambda e: (e["date"], severity_order.get(e["severity"], 9)))

    logger.info(f"Structure change detection: {len(events)} events found")
    return events


# ═══════════════════════════════════════════════════════════════════
# 5. Broker Momentum Time Series
# ═══════════════════════════════════════════════════════════════════

def calc_broker_momentum_timeseries(
    historical_data: dict[str, pd.DataFrame],
    top_n: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Track each broker's net position and rank over time.

    For each broker appearing in the data, computes a time series of:
      - net_position = long_pos - short_pos
      - delta_np = day-over-day change in net position
      - rank_long, rank_short = rank among all brokers
      - cumulative_dnp = rolling sum of delta_np over BROKER_RANK_LOOKBACK days

    Parameters
    ----------
    historical_data : dict[str, pd.DataFrame]
        {date_str: DataFrame} from load_historical_positions().
    top_n : int
        Only track brokers that appear in the top-N (by total position)
        at least once.

    Returns
    -------
    dict[str, pd.DataFrame]
        {broker_name: DataFrame with columns [date, net_pos, delta_np,
         rank_long, rank_short, long_pos, short_pos]}
        Only includes brokers who appear in top_n at least once.
        Empty dict if no data.
    """
    if not historical_data:
        return {}

    # Collect all (date, broker) pairs
    all_records = []
    for date_str, df in sorted(historical_data.items()):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            broker = row.get("broker", "")
            if not broker:
                continue
            all_records.append({
                "date": date_str,
                "broker": broker,
                "long_pos": int(row.get("long_pos", 0)),
                "short_pos": int(row.get("short_pos", 0)),
                "long_change": int(row.get("long_change", 0)),
                "short_change": int(row.get("short_change", 0)),
                "volume": int(row.get("volume", 0)),
            })

    if not all_records:
        return {}

    full_df = pd.DataFrame(all_records)
    full_df["date"] = pd.to_datetime(
        full_df["date"], format="%Y%m%d", errors="coerce"
    )
    full_df["net_pos"] = full_df["long_pos"] - full_df["short_pos"]

    # Find brokers that appear in top_n by abs(net_pos) at least once
    top_brokers_set = set()
    for date_val, grp in full_df.groupby("date"):
        top_by_total = grp.nlargest(
            top_n, "long_pos"
        )["broker"].tolist()
        top_by_short = grp.nlargest(
            top_n, "short_pos"
        )["broker"].tolist()
        top_brokers_set.update(top_by_total)
        top_brokers_set.update(top_by_short)
        # Also include brokers with largest absolute net position
        grp_copy = grp.copy()
        grp_copy["abs_net"] = grp_copy["net_pos"].abs()
        top_by_net = grp_copy.nlargest(top_n, "abs_net")["broker"].tolist()
        top_brokers_set.update(top_by_net)

    # Filter to tracked brokers
    tracked = full_df[full_df["broker"].isin(top_brokers_set)].copy()

    # Compute ranks per date
    tracked["rank_long"] = tracked.groupby("date")["long_pos"].rank(
        ascending=False, method="min"
    )
    tracked["rank_short"] = tracked.groupby("date")["short_pos"].rank(
        ascending=False, method="min"
    )

    # Compute delta_np per broker
    result = {}
    for broker, grp in tracked.groupby("broker"):
        grp_sorted = grp.sort_values("date")
        grp_sorted["delta_np"] = grp_sorted["net_pos"].diff().fillna(0).astype(int)
        # Cumulative delta over lookback
        grp_sorted["cumulative_dnp"] = (
            grp_sorted["delta_np"]
            .rolling(window=BROKER_RANK_LOOKBACK, min_periods=1)
            .sum()
        )
        result[broker] = grp_sorted[
            ["date", "net_pos", "delta_np", "cumulative_dnp",
             "rank_long", "rank_short", "long_pos", "short_pos"]
        ].reset_index(drop=True)

    logger.info(
        f"Broker momentum: {len(result)} brokers tracked "
        f"across {len(historical_data)} days"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# 6. Top Broker Rank Stability
# ═══════════════════════════════════════════════════════════════════

def calc_rank_stability(
    cr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure how stable the top broker composition is over time.

    For each date, compares the set of top-N brokers to the previous
    date's set. A "rank shake-up" occurs when the overlap drops.

    Parameters
    ----------
    cr_df : pd.DataFrame
        Output of calc_cr_timeseries(). Must have top_long_brokers
        and top_short_brokers columns.

    Returns
    -------
    pd.DataFrame
        Columns: date, long_overlap, short_overlap, long_stability,
                 short_stability, long_shakeup, short_shakeup.
        stability = Jaccard similarity between consecutive days' top sets.
    """
    if cr_df is None or cr_df.empty or len(cr_df) < 2:
        return pd.DataFrame()

    df = cr_df.copy()
    rows = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_long = set(prev.get("top_long_brokers", []) or [])
        curr_long = set(curr.get("top_long_brokers", []) or [])
        prev_short = set(prev.get("top_short_brokers", []) or [])
        curr_short = set(curr.get("top_short_brokers", []) or [])

        # Jaccard similarity
        def _jaccard(a, b):
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        long_stab = round(_jaccard(prev_long, curr_long), 3)
        short_stab = round(_jaccard(prev_short, curr_short), 3)

        rows.append({
            "date": curr["date"],
            "long_overlap": len(prev_long & curr_long),
            "short_overlap": len(prev_short & curr_short),
            "long_stability": long_stab,
            "short_stability": short_stab,
            "long_shakeup": long_stab < 0.5,   # >50% of top-N changed
            "short_shakeup": short_stab < 0.5,
            "new_long_brokers": sorted(curr_long - prev_long),
            "new_short_brokers": sorted(curr_short - prev_short),
            "dropped_long_brokers": sorted(prev_long - curr_long),
            "dropped_short_brokers": sorted(prev_short - curr_short),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 7. Broker Defection Detection
# ═══════════════════════════════════════════════════════════════════

def detect_broker_defections(
    broker_ts: dict[str, pd.DataFrame],
    min_history_days: int = 3,
) -> list[dict]:
    """
    Detect when key brokers change their directional stance.

    Event types:
      - "flip_long_to_short": net_pos crosses from > 0 to < 0
      - "flip_short_to_long": net_pos crosses from < 0 to > 0
      - "long_unwind": net long position shrinks significantly
      - "short_unwind": net short position shrinks significantly
      - "momentum_reversal": cumulative ΔNP sign flips (was adding → now cutting)
      - "surrender": dominant broker suddenly cuts >50% of position

    Parameters
    ----------
    broker_ts : dict[str, pd.DataFrame]
        Output of calc_broker_momentum_timeseries().
    min_history_days : int
        Minimum consecutive days a broker must appear to be analyzed.

    Returns
    -------
    list[dict]
        Each event: {
            "date", "broker", "type", "severity", "detail",
            "metrics": {net_pos_before, net_pos_after, delta, ...}
        }
        Sorted by date, then severity.
    """
    from src.config import (
        DEFECTION_MOMENTUM_WINDOW,
        DEFECTION_SHRINK_THRESHOLD,
    )

    if not broker_ts:
        return []

    events = []

    for broker, ts_df in broker_ts.items():
        if ts_df is None or ts_df.empty or len(ts_df) < min_history_days:
            continue

        df = ts_df.sort_values("date").reset_index(drop=True)

        # ── a) Direction flip: net_pos crosses zero ──
        df["net_sign"] = np.sign(df["net_pos"])
        df["prev_sign"] = df["net_sign"].shift(1)

        # Long → Short (positive → negative)
        flip_l2s = (df["prev_sign"] > 0) & (df["net_sign"] < 0)
        for idx in df.index[flip_l2s]:
            row = df.loc[idx]
            prev = df.loc[idx - 1] if idx > 0 else row
            events.append({
                "date": _fmt_date(row["date"]),
                "broker": broker,
                "type": "flip_long_to_short",
                "severity": "high",
                "detail": (
                    f"**{broker}** 从净多头翻为净空头。"
                    f"净持仓 {prev['net_pos']:+} → {row['net_pos']:+}，"
                    f"Δ = {row['delta_np']:+}。"
                ),
                "metrics": {
                    "net_pos_before": int(prev["net_pos"]),
                    "net_pos_after": int(row["net_pos"]),
                    "delta_np": int(row["delta_np"]),
                },
            })

        # Short → Long (negative → positive)
        flip_s2l = (df["prev_sign"] < 0) & (df["net_sign"] > 0)
        for idx in df.index[flip_s2l]:
            row = df.loc[idx]
            prev = df.loc[idx - 1] if idx > 0 else row
            events.append({
                "date": _fmt_date(row["date"]),
                "broker": broker,
                "type": "flip_short_to_long",
                "severity": "high",
                "detail": (
                    f"**{broker}** 从净空头翻为净多头。"
                    f"净持仓 {prev['net_pos']:+} → {row['net_pos']:+}，"
                    f"Δ = {row['delta_np']:+}。"
                ),
                "metrics": {
                    "net_pos_before": int(prev["net_pos"]),
                    "net_pos_after": int(row["net_pos"]),
                    "delta_np": int(row["delta_np"]),
                },
            })

        # ── b) Significant unwind ──
        df["net_pos_abs_change"] = df["net_pos"].abs().diff().abs()
        df["prev_net_abs"] = df["net_pos"].abs().shift(1)

        long_mask = df["net_pos"] > 0
        short_mask = df["net_pos"] < 0
        shrink_mask = (
            (df["net_pos_abs_change"] / df["prev_net_abs"].replace(0, np.nan))
            > DEFECTION_SHRINK_THRESHOLD
        )

        # Long unwind
        long_unwind = long_mask & shrink_mask & (df["delta_np"] < 0)
        for idx in df.index[long_unwind]:
            row = df.loc[idx]
            prev = df.loc[idx - 1] if idx > 0 else row
            pct = abs(row["delta_np"]) / max(abs(prev["net_pos"]), 1)
            events.append({
                "date": _fmt_date(row["date"]),
                "broker": broker,
                "type": "long_unwind",
                "severity": "high" if pct > 0.5 else "medium",
                "detail": (
                    f"**{broker}** 多头大幅减仓。"
                    f"净多 {prev['net_pos']:+} → {row['net_pos']:+} "
                    f"（减仓 {abs(row['delta_np']):,}，幅度 {pct:.0%}）。"
                ),
                "metrics": {
                    "net_pos_before": int(prev["net_pos"]),
                    "net_pos_after": int(row["net_pos"]),
                    "delta_np": int(row["delta_np"]),
                    "shrink_pct": round(pct, 3),
                },
            })

        # Short unwind
        short_unwind = short_mask & shrink_mask & (df["delta_np"] > 0)
        for idx in df.index[short_unwind]:
            row = df.loc[idx]
            prev = df.loc[idx - 1] if idx > 0 else row
            pct = abs(row["delta_np"]) / max(abs(prev["net_pos"]), 1)
            events.append({
                "date": _fmt_date(row["date"]),
                "broker": broker,
                "type": "short_unwind",
                "severity": "high" if pct > 0.5 else "medium",
                "detail": (
                    f"**{broker}** 空头大幅减仓。"
                    f"净空 {prev['net_pos']:+} → {row['net_pos']:+} "
                    f"（回补 {row['delta_np']:+,}，幅度 {pct:.0%}）。"
                ),
                "metrics": {
                    "net_pos_before": int(prev["net_pos"]),
                    "net_pos_after": int(row["net_pos"]),
                    "delta_np": int(row["delta_np"]),
                    "shrink_pct": round(pct, 3),
                },
            })

        # ── c) Momentum reversal ──
        if len(df) >= DEFECTION_MOMENTUM_WINDOW:
            df["cum_sign"] = np.sign(df["cumulative_dnp"])
            df["prev_cum_sign"] = df["cum_sign"].shift(1)

            # Cumulative momentum flips sign
            mom_reverse = (
                (df["prev_cum_sign"] != 0) &
                (df["cum_sign"] != 0) &
                (df["prev_cum_sign"] != df["cum_sign"])
            )
            for idx in df.index[mom_reverse]:
                row = df.loc[idx]
                prev_sign = "加多" if row["prev_cum_sign"] > 0 else "加空"
                cur_sign = "加多" if row["cum_sign"] > 0 else "加空"
                events.append({
                    "date": _fmt_date(row["date"]),
                    "broker": broker,
                    "type": "momentum_reversal",
                    "severity": "medium",
                    "detail": (
                        f"**{broker}** 持仓动量逆转："
                        f"此前持续{prev_sign}，现已转为{cur_sign}。"
                        f"累计 ΔNP = {row['cumulative_dnp']:+}。"
                    ),
                    "metrics": {
                        "cumulative_dnp": int(row["cumulative_dnp"]),
                        "delta_np_today": int(row["delta_np"]),
                    },
                })

    # Sort by date then severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    events.sort(key=lambda e: (e["date"], severity_order.get(e["severity"], 9)))

    logger.info(f"Defection detection: {len(events)} events across {len(broker_ts)} brokers")
    return events


def _fmt_date(date_val) -> str:
    """Format a date value to string YYYY-MM-DD."""
    if hasattr(date_val, "date"):
        return str(date_val.date())
    return str(date_val)[:10] if len(str(date_val)) >= 10 else str(date_val)


# ═══════════════════════════════════════════════════════════════════
# 8. Broker-Price Correlation — Smart Money Detection
# ═══════════════════════════════════════════════════════════════════

def calc_broker_price_correlation(
    broker_ts: dict[str, pd.DataFrame],
    price_df: pd.DataFrame,
    lag_days: list[int] = None,
    min_overlap: int = None,
) -> pd.DataFrame:
    """
    Correlate each broker's ΔNP with subsequent price changes.

    For each broker and each lag window, computes:
      - Pearson r between ΔNP(t) and price_change(t+lag)
      - Spearman rank correlation (more robust to outliers)
      - Hit rate: % of times ΔNP sign matches future price direction
      - Average impact: mean price change following positive ΔNP vs negative ΔNP

    Parameters
    ----------
    broker_ts : dict[str, pd.DataFrame]
        Output of calc_broker_momentum_timeseries().
    price_df : pd.DataFrame
        Daily price data. Must have 'date' and 'close' columns.
    lag_days : list[int], optional
        Lag windows to test. Default [1, 3, 5].
    min_overlap : int, optional
        Minimum overlapping data points. Defaults to PRICE_CORR_MIN_DAYS.

    Returns
    -------
    pd.DataFrame
        Columns: broker, lag, pearson_r, spearman_r, hit_rate,
                 avg_impact, total_obs, is_smart_money, best_lag.
        Sorted by |pearson_r| descending at best_lag.
        Each broker appears once (at its best lag).
    """
    from src.config import (
        PRICE_CORR_LAG_DAYS,
        PRICE_CORR_MIN_DAYS,
        SMART_MONEY_CORR_THRESHOLD,
    )

    if lag_days is None:
        lag_days = PRICE_CORR_LAG_DAYS
    if min_overlap is None:
        min_overlap = PRICE_CORR_MIN_DAYS

    if not broker_ts or price_df is None or price_df.empty:
        logger.warning("calc_broker_price_correlation: insufficient data")
        return pd.DataFrame()

    # Prepare price data
    price = price_df.copy()
    if "date" in price.columns:
        price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.sort_values("date").reset_index(drop=True)
    price["price_change"] = price["close"].pct_change()
    price["price_change_fwd"] = price["close"].shift(-1) / price["close"] - 1

    results = []

    for broker, ts_df in broker_ts.items():
        if ts_df is None or ts_df.empty or len(ts_df) < min_overlap:
            continue

        df = ts_df.sort_values("date").reset_index(drop=True)
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

        # Merge broker ΔNP with price data by date
        merged = pd.merge(
            df[["date_dt", "delta_np", "net_pos"]],
            price[["date", "close", "price_change", "price_change_fwd"]],
            left_on="date_dt", right_on="date", how="inner",
        )

        if len(merged) < min_overlap:
            continue

        best_lag = None
        best_pearson = -1
        best_row = None

        for lag in lag_days:
            merged["future_return"] = merged["close"].shift(-lag) / merged["close"] - 1
            valid = merged.dropna(subset=["delta_np", "future_return"])

            if len(valid) < min_overlap:
                continue

            # Pearson correlation
            pearson_r = valid["delta_np"].corr(valid["future_return"])

            # Spearman rank correlation
            spearman_r = valid["delta_np"].corr(valid["future_return"], method="spearman")

            # Hit rate: does ΔNP sign predict future return sign?
            same_sign = (
                (np.sign(valid["delta_np"]) == np.sign(valid["future_return"]))
                & (valid["delta_np"] != 0)
            )
            hit_rate = same_sign.sum() / max(len(same_sign), 1)

            # Average impact
            pos_dnp = valid[valid["delta_np"] > 0]
            neg_dnp = valid[valid["delta_np"] < 0]
            avg_impact_pos = pos_dnp["future_return"].mean() if len(pos_dnp) > 0 else 0
            avg_impact_neg = neg_dnp["future_return"].mean() if len(neg_dnp) > 0 else 0
            avg_impact = avg_impact_pos - avg_impact_neg

            row_data = {
                "broker": broker,
                "lag": lag,
                "pearson_r": round(pearson_r, 4) if not np.isnan(pearson_r) else 0.0,
                "spearman_r": round(spearman_r, 4) if not np.isnan(spearman_r) else 0.0,
                "hit_rate": round(hit_rate, 3),
                "avg_impact": round(avg_impact, 6),
                "total_obs": len(valid),
            }
            results.append(row_data)

            # Track best lag for this broker
            abs_r = abs(pearson_r) if not np.isnan(pearson_r) else 0
            if abs_r > best_pearson:
                best_pearson = abs_r
                best_lag = lag
                best_row = row_data

    if not results:
        return pd.DataFrame()

    all_df = pd.DataFrame(results)

    # Mark smart money: brokers whose |pearson_r| exceeds threshold
    all_df["is_smart_money"] = all_df["pearson_r"].abs() >= SMART_MONEY_CORR_THRESHOLD

    # Determine the best lag for each broker
    best_lags = (
        all_df.groupby("broker")
        .apply(lambda g: g.loc[g["pearson_r"].abs().idxmax()], include_groups=False)
        .reset_index()
    )
    best_lags = best_lags.sort_values(
        "pearson_r", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)

    logger.info(
        f"Price correlation: {len(best_lags)} brokers analyzed, "
        f"{best_lags['is_smart_money'].sum()} flagged as smart money"
    )
    return best_lags


def identify_smart_money(
    corr_df: pd.DataFrame,
) -> dict[str, list[dict]]:
    """
    Categorize brokers by their price prediction characteristics.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of calc_broker_price_correlation().

    Returns
    -------
    dict
        {
            "leading_long": [{broker, r, hit_rate, lag}, ...],   # predicts UP
            "leading_short": [{broker, r, hit_rate, lag}, ...],  # predicts DOWN
            "contrarian": [{broker, r, hit_rate, lag}, ...],     # anti-correlated
            "noisy": [{broker, ...}],                             # no clear signal
        }
    """
    from src.config import SMART_MONEY_CORR_THRESHOLD

    if corr_df is None or corr_df.empty:
        return {"leading_long": [], "leading_short": [], "contrarian": [], "noisy": []}

    threshold = SMART_MONEY_CORR_THRESHOLD
    result = {"leading_long": [], "leading_short": [], "contrarian": [], "noisy": []}

    for _, row in corr_df.iterrows():
        broker_info = {
            "broker": row["broker"],
            "pearson_r": row["pearson_r"],
            "spearman_r": row["spearman_r"],
            "hit_rate": row["hit_rate"],
            "lag": int(row["lag"]),
            "avg_impact": row["avg_impact"],
        }

        r = row["pearson_r"]
        if abs(r) < threshold:
            result["noisy"].append(broker_info)
        elif r > 0:
            result["leading_long"].append(broker_info)
        else:
            result["contrarian"].append(broker_info)

    # Sort each group by |r| descending
    for key in result:
        result[key].sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    return result


# ═══════════════════════════════════════════════════════════════════
# 9. Position Turnover Rate — Churn Detector
# ═══════════════════════════════════════════════════════════════════

def calc_turnover_ratio(
    historical_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate daily position turnover (churn) rate.

    Turnover = Σ|long_change_i| / Σ(long_pos_i)
    This measures: what fraction of total long positions were
    actively changed (added or reduced) today?

    A spike in turnover signals aggressive repositioning —
    often precedes a structural regime change.

    Also computes:
      - net_turnover: (Σ long_change - Σ short_change) / Σ positions
        Positive = net buying pressure, Negative = net selling pressure
      - turnover_asymmetry: |long_turnover - short_turnover|
        Large asymmetry = one-sided repositioning

    Parameters
    ----------
    historical_data : dict[str, pd.DataFrame]
        {date_str: DataFrame} from load_historical_positions().
        DataFrames must have long_change and short_change columns.

    Returns
    -------
    pd.DataFrame
        Columns: date, long_turnover, short_turnover, total_turnover,
                 net_turnover, asymmetry.
    """
    if not historical_data:
        return pd.DataFrame()

    from src.config import TURNOVER_HIGH_THRESHOLD, TURNOVER_SPIKE_THRESHOLD

    rows = []
    for date_str, df in sorted(historical_data.items()):
        if df is None or df.empty:
            continue

        has_long_chg = "long_change" in df.columns
        has_short_chg = "short_change" in df.columns

        if not has_long_chg and not has_short_chg:
            continue

        total_long = df["long_pos"].sum() if "long_pos" in df.columns else 0
        total_short = df["short_pos"].sum() if "short_pos" in df.columns else 0

        long_churn = (
            df["long_change"].abs().sum() / total_long
            if has_long_chg and total_long > 0 else 0.0
        )
        short_churn = (
            df["short_change"].abs().sum() / total_short
            if has_short_chg and total_short > 0 else 0.0
        )

        total_positions = total_long + total_short
        total_turnover = (
            (df["long_change"].abs().sum() + df["short_change"].abs().sum())
            / total_positions
            if has_long_chg and has_short_chg and total_positions > 0
            else max(long_churn, short_churn)
        )

        net_turnover = (
            (df["long_change"].sum() - df["short_change"].sum()) / total_positions
            if total_positions > 0 else 0.0
        )

        asymmetry = abs(long_churn - short_churn)

        rows.append({
            "date": date_str,
            "long_turnover": round(long_churn, 4),
            "short_turnover": round(short_churn, 4),
            "total_turnover": round(total_turnover, 4),
            "net_turnover": round(net_turnover, 4),
            "asymmetry": round(asymmetry, 4),
            "is_high_churn": total_turnover >= TURNOVER_HIGH_THRESHOLD,
            "is_spike": total_turnover >= TURNOVER_SPIKE_THRESHOLD,
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["date"] = pd.to_datetime(
        df_out["date"], format="%Y%m%d", errors="coerce"
    )
    return df_out.sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 10. New Entrant Detection — Structure Dilution
# ═══════════════════════════════════════════════════════════════════

def detect_new_entrants(
    historical_data: dict[str, pd.DataFrame],
    top_n: int = None,
) -> pd.DataFrame:
    """
    Track how many new brokers enter the top-N ranking each day.

    When many new entrants appear, the existing power structure is
    being diluted — a leading indicator of structural loosening.

    For each day, compares the set of top-N brokers (by total position)
    to the union of all brokers seen in the previous LOOKBACK days.

    Parameters
    ----------
    historical_data : dict[str, pd.DataFrame]
        {date_str: DataFrame} from load_historical_positions().
    top_n : int
        How many top brokers to track. Defaults to NEW_ENTRANT_TOP_N.

    Returns
    -------
    pd.DataFrame
        Columns: date, total_brokers, top_n_count, new_entrants,
                 new_entrant_ratio, retained_ratio, entrant_names,
                 exited_names.
    """
    from src.config import NEW_ENTRANT_LOOKBACK, NEW_ENTRANT_TOP_N

    if top_n is None:
        top_n = NEW_ENTRANT_TOP_N
    lookback = NEW_ENTRANT_LOOKBACK

    if not historical_data or len(historical_data) < 2:
        return pd.DataFrame()

    sorted_dates = sorted(historical_data.keys())
    rows = []

    # Rolling set of brokers seen in the lookback window
    seen_brokers = set()

    for i, date_str in enumerate(sorted_dates):
        df = historical_data[date_str]
        if df is None or df.empty:
            continue

        # Today's top-N by total position
        df_copy = df.copy()
        df_copy["total_pos"] = (
            df_copy.get("long_pos", 0).fillna(0) +
            df_copy.get("short_pos", 0).fillna(0)
        )
        top_n_df = df_copy.nlargest(min(top_n, len(df_copy)), "total_pos")
        today_top = set(top_n_df["broker"].tolist())

        # New entrants: in today's top-N but not in seen set
        new_brokers = today_top - seen_brokers
        # Exited: were in seen set but not in today's top-N
        exited_brokers = seen_brokers - today_top

        new_ratio = len(new_brokers) / max(len(today_top), 1)

        rows.append({
            "date": date_str,
            "total_brokers": len(df_copy),
            "top_n_count": len(today_top),
            "new_entrants": len(new_brokers),
            "new_entrant_ratio": round(new_ratio, 3),
            "retained_ratio": round(1.0 - new_ratio, 3),
            "entrant_names": sorted(new_brokers),
            "exited_names": sorted(exited_brokers),
            "seen_pool_size": len(seen_brokers),
        })

        # Add today's brokers to the seen pool
        seen_brokers.update(today_top)

        # Limit seen pool to most recent N days worth of brokers
        if i >= lookback:
            # Rebuild seen pool from last 'lookback' days
            seen_brokers = set()
            for j in range(max(0, i - lookback + 1), i + 1):
                prev_df = historical_data.get(sorted_dates[j])
                if prev_df is not None and not prev_df.empty:
                    prev_copy = prev_df.copy()
                    prev_copy["total_pos"] = (
                        prev_copy.get("long_pos", 0).fillna(0) +
                        prev_copy.get("short_pos", 0).fillna(0)
                    )
                    prev_top = prev_copy.nlargest(
                        min(top_n, len(prev_copy)), "total_pos"
                    )
                    seen_brokers.update(prev_top["broker"].tolist())

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["date"] = pd.to_datetime(
        df_out["date"], format="%Y%m%d", errors="coerce"
    )
    return df_out.sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 11. Bull-Bear Divergence Index
# ═══════════════════════════════════════════════════════════════════

def calc_divergence_index(
    cr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a composite bull-bear divergence index.

    Divergence Index = 1 − |CR_long − CR_short|
    Range: 0 (total dominance by one side) to 1 (perfect balance).

    Additional metrics:
      - force_balance: signed version (−1 = short dominant, +1 = long dominant)
      - convergence: is the gap shrinking? (True = forces converging)
      - balance_trend: "diverging" | "converging" | "stable"

    Parameters
    ----------
    cr_df : pd.DataFrame
        Output of calc_cr_timeseries(). Must have cr_long, cr_short, cr_diff.

    Returns
    -------
    pd.DataFrame
        Columns: date, divergence_index, force_balance, cr_diff_abs,
                 cr_diff_change, convergence, balance_trend.
    """
    if cr_df is None or cr_df.empty:
        return pd.DataFrame()

    from src.config import DIVERGENCE_SHRINK_THRESHOLD

    df = cr_df[["date", "cr_long", "cr_short", "cr_diff"]].copy()

    df["cr_diff_abs"] = df["cr_diff"].abs()
    df["divergence_index"] = round(1.0 - df["cr_diff_abs"], 4)
    df["force_balance"] = round(df["cr_diff"], 4)

    # Day-over-day change in the gap
    df["cr_diff_change"] = df["cr_diff_abs"].diff().fillna(0)

    # Convergence: gap is shrinking
    df["convergence"] = df["cr_diff_change"] < -DIVERGENCE_SHRINK_THRESHOLD

    # Balance trend
    def _trend(row):
        change = row.get("cr_diff_change", 0)
        if change > DIVERGENCE_SHRINK_THRESHOLD:
            return "diverging"
        elif change < -DIVERGENCE_SHRINK_THRESHOLD:
            return "converging"
        return "stable"

    df["balance_trend"] = df.apply(_trend, axis=1)

    # HHI divergence: are HHI and CR moving in opposite directions?
    if "hhi_long" in cr_df.columns:
        df["hhi_long"] = cr_df["hhi_long"]
        df["hhi_short"] = cr_df["hhi_short"]
        df["hhi_long_change"] = df["hhi_long"].diff().fillna(0)
        df["hhi_short_change"] = df["hhi_short"].diff().fillna(0)
        df["hhi_cr_divergence"] = (
            (df["hhi_long_change"] * df["cr_long"].diff().fillna(0)) < 0
        )

    return df


# ═══════════════════════════════════════════════════════════════════
# 12. Concentration Profile — Full Distribution
# ═══════════════════════════════════════════════════════════════════

def calc_concentration_profile(
    df: pd.DataFrame,
) -> dict:
    """
    Compute the full concentration distribution for a single day.

    Goes beyond CR (top-3) and HHI (entire distribution) to give
    a multi-level view:
      - CR_1, CR_3, CR_5, CR_10 (top-1, top-3, top-5, top-10)
      - Gini coefficient of position shares
      - Top-to-bottom ratio (T10 / B10)

    Parameters
    ----------
    df : pd.DataFrame
        Single day's position data with broker, long_pos, short_pos.

    Returns
    -------
    dict
        {cr_1_long, cr_3_long, cr_5_long, cr_10_long,
         cr_1_short, cr_3_short, cr_5_short, cr_10_short,
         gini_long, gini_short, top_bottom_ratio_long,
         top_bottom_ratio_short, broker_count}
    """
    if df is None or df.empty:
        return {}

    result = {"broker_count": len(df)}

    for side, col in [("long", "long_pos"), ("short", "short_pos")]:
        if col not in df.columns:
            continue

        total = df[col].sum()
        if total <= 0:
            continue

        shares = df[col].sort_values(ascending=False) / total
        cum_shares = shares.cumsum()

        for n in [1, 3, 5, 10]:
            key = f"cr_{n}_{side}"
            if n <= len(cum_shares):
                result[key] = round(float(cum_shares.iloc[n - 1]), 4)
            else:
                result[key] = round(float(cum_shares.iloc[-1]), 4)

        # Gini coefficient
        sorted_shares = shares.values
        n_brokers = len(sorted_shares)
        if n_brokers > 1:
            ranks = np.arange(1, n_brokers + 1)
            gini = (
                (2 * np.sum(ranks * sorted_shares) - (n_brokers + 1))
                / n_brokers
            )
            result[f"gini_{side}"] = round(max(0.0, float(gini)), 4)
        else:
            result[f"gini_{side}"] = 0.0

        # Top/Bottom ratio
        if n_brokers >= 10:
            top10_sum = sorted_shares[:10].sum()
            bottom10_sum = sorted_shares[-10:].sum()
            ratio = top10_sum / bottom10_sum if bottom10_sum > 0 else float("inf")
            result[f"top_bottom_ratio_{side}"] = round(float(ratio), 1)
        else:
            result[f"top_bottom_ratio_{side}"] = float("inf") if n_brokers > 1 else 1.0

    return result


def calc_concentration_profile_timeseries(
    historical_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute concentration profile over time.

    Parameters
    ----------
    historical_data : dict[str, pd.DataFrame]
        {date_str: DataFrame} from load_historical_positions().

    Returns
    -------
    pd.DataFrame
        Columns: date, cr_1_long, cr_3_long, cr_5_long, cr_10_long,
                 cr_1_short, cr_3_short, cr_5_short, cr_10_short,
                 gini_long, gini_short, top_bottom_ratio_long, ...
    """
    if not historical_data:
        return pd.DataFrame()

    rows = []
    for date_str, df in sorted(historical_data.items()):
        profile = calc_concentration_profile(df)
        if profile:
            profile["date"] = date_str
            rows.append(profile)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["date"] = pd.to_datetime(
        df_out["date"], format="%Y%m%d", errors="coerce"
    )
    return df_out.sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 13. Composite Structure Health Score
# ═══════════════════════════════════════════════════════════════════

def calc_structure_score(
    cr_trend_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    entrants_df: pd.DataFrame,
    divergence_df: pd.DataFrame,
    defections: list,
    stability_df: pd.DataFrame,
    symbol: str = "",
) -> dict:
    """Composite 0–100 score: healthy/caution/unstable/critical."""
    components = {}
    total = 0

    # 1. CR Stability (25)
    if cr_trend_df is not None and not cr_trend_df.empty:
        r = cr_trend_df.iloc[-1]
        sl, ss = r.get("structure_long", "stable"), r.get("structure_short", "stable")
        if sl == "loosening" and ss == "loosening":
            cs, cd = 5, "多空集中度双双松动"
        elif sl == "loosening" or ss == "loosening":
            cs, cd = 13, f"{'多头' if sl == 'loosening' else '空头'}集中度松动"
        elif sl == "tightening" or ss == "tightening":
            cs, cd = 17, f"{'多头' if sl == 'tightening' else '空头'}集中度收紧"
        else:
            cs, cd = 25, "集中度趋势稳定"
    else:
        cs, cd = 12, "无数据"
    components["cr_stability"] = {"score": cs, "max": 25, "detail": cd}; total += cs

    # 2. Turnover (20)
    if turnover_df is not None and not turnover_df.empty:
        tv = turnover_df.iloc[-1].get("total_turnover", 0) or 0
        if tv < 0.15:    ts, td = 20, f"低换手({tv:.1%})"
        elif tv < 0.30:  ts, td = 15, f"正常({tv:.1%})"
        elif tv < 0.50:  ts, td = 8, f"偏高({tv:.1%})"
        else:            ts, td = 2, f"剧烈({tv:.1%})！"
    else:
        ts, td = 10, "无数据"
    components["turnover"] = {"score": ts, "max": 20, "detail": td}; total += ts

    # 3. New Entrants (15)
    if entrants_df is not None and not entrants_df.empty:
        r = entrants_df.iloc[-1]
        nr, nc = r.get("new_entrant_ratio", 0) or 0, int(r.get("new_entrants", 0) or 0)
        if nr == 0:       es, ed = 15, "无新进入者"
        elif nr < 0.15:   es, ed = 10, f"少量({nc}席)"
        elif nr < 0.30:   es, ed = 6, f"较多({nc}席)"
        else:             es, ed = 2, f"大量({nc}席)！"
    else:
        es, ed = 7, "无数据"
    components["new_entrants"] = {"score": es, "max": 15, "detail": ed}; total += es

    # 4. Divergence (15)
    if divergence_df is not None and not divergence_df.empty:
        trend = divergence_df.iloc[-1].get("balance_trend", "stable")
        if trend == "stable":       ds, dd = 15, "力量对比稳定"
        elif trend == "converging": ds, dd = 7, "力量收敛中"
        else:                       ds, dd = 5, "力量拉大中"
    else:
        ds, dd = 7, "无数据"
    components["divergence"] = {"score": ds, "max": 15, "detail": dd}; total += ds

    # 5. Defections (15)
    if defections:
        dates = sorted(set(e["date"] for e in defections))
        recent = [e for e in defections if e["date"] >= _days_ago(dates[-1], 5)] if dates else defections
        hc = sum(1 for e in recent if e.get("severity") == "high")
        tc = len(recent)
        if tc == 0:       fs, fd = 15, "无叛变"
        elif tc <= 2:     fs, fd = 10, f"{tc}起(高危{hc})"
        elif tc <= 5:     fs, fd = 5, f"{tc}起(高危{hc})"
        else:             fs, fd = 1, f"{tc}起(高危{hc})！"
    else:
        fs, fd = 7, "无数据"
    components["defections"] = {"score": fs, "max": 15, "detail": fd}; total += fs

    # 6. Rank Stability (10)
    if stability_df is not None and not stability_df.empty:
        r = stability_df.iloc[-1]
        avg = ((r.get("long_stability", 0.5) or 0.5) + (r.get("short_stability", 0.5) or 0.5)) / 2
        if avg >= 0.8:    rs, rd = 10, f"高度稳定({avg:.0%})"
        elif avg >= 0.5:  rs, rd = 7, f"基本稳定({avg:.0%})"
        else:             rs, rd = 3, f"震荡({avg:.0%})"
    else:
        rs, rd = 5, "无数据"
    components["rank_stability"] = {"score": rs, "max": 10, "detail": rd}; total += rs

    # Status
    if total >= 80:      status, summary = "healthy", "持仓结构健康，席位格局稳定。"
    elif total >= 60:    status, summary = "caution", "部分指标出现松动，值得关注。"
    elif total >= 40:    status, summary = "unstable", "持仓结构明显松动，建议密切关注。"
    else:                status, summary = "critical", "结构剧烈变动！席位格局可能重构。"

    date_str = ""
    for df in [cr_trend_df, turnover_df, entrants_df, divergence_df]:
        if df is not None and not df.empty and "date" in df.columns:
            date_str = _fmt_date(df["date"].iloc[-1]); break

    return {"symbol": symbol, "date": date_str, "score": total,
            "status": status, "components": components, "summary": summary}


def _days_ago(date_str: str, days: int) -> str:
    try:
        parts = date_str.split("-")
        if len(parts) == 3:
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
            return (dt - timedelta(days=days)).strftime("%Y-%m-%d")
    except Exception:
        pass
    return date_str


def calc_all_symbols_scores(
    historical_by_symbol: dict,
    broker_ts_by_symbol: dict,
    defections_by_symbol: dict,
    turnover_by_symbol: dict,
    entrants_by_symbol: dict,
) -> list[dict]:
    """Compute structure scores for all symbols, sorted worst-first."""
    results = []
    for symbol in historical_by_symbol:
        hist = historical_by_symbol.get(symbol, {})
        if not hist: continue
        cr_df = calc_cr_timeseries(hist)
        cr_df = calc_cr_trend(cr_df)
        div_df = calc_divergence_index(cr_df)
        stb_df = calc_rank_stability(cr_df)
        score = calc_structure_score(
            cr_trend_df=cr_df, turnover_df=turnover_by_symbol.get(symbol),
            entrants_df=entrants_by_symbol.get(symbol), divergence_df=div_df,
            defections=defections_by_symbol.get(symbol, []),
            stability_df=stb_df, symbol=symbol,
        )
        results.append(score)
    results.sort(key=lambda r: r["score"])
    return results


# ═══════════════════════════════════════════════════════════════════
# 14. Structure Alerts
# ═══════════════════════════════════════════════════════════════════

def detect_structure_alerts(
    score: dict, prev_score: Optional[dict] = None,
) -> list[dict]:
    """Generate alerts from structure score and changes."""
    alerts = []
    symbol = score.get("symbol", "?")
    total = score["score"]

    if total < 40:
        alerts.append({"type": "structure_critical", "symbol": symbol,
                       "severity": "high", "score": total,
                       "detail": f"{symbol.upper()} 结构评分{total}/100，危险区。{score['summary']}"})
    elif total < 60:
        alerts.append({"type": "structure_unstable", "symbol": symbol,
                       "severity": "medium", "score": total,
                       "detail": f"{symbol.upper()} 结构评分{total}/100，不稳定区。{score['summary']}"})

    if prev_score and prev_score.get("score", 100) - total >= 15:
        drop = prev_score["score"] - total
        alerts.append({"type": "structure_score_drop", "symbol": symbol,
                       "severity": "high", "score": total, "score_drop": drop,
                       "detail": f"{symbol.upper()} 评分骤降{drop}分({prev_score['score']}→{total})。"})

    for cn, c in score.get("components", {}).items():
        if c["score"] <= 1:
            alerts.append({"type": "structure_component_breakdown", "symbol": symbol,
                           "severity": "medium", "component": cn,
                           "detail": f"{symbol.upper()} {cn}触底({c['score']}/{c['max']})。{c['detail']}"})

    return alerts


# ═══════════════════════════════════════════════════════════════════
# 15. Broker Scatter Data — for smart-money visualization
# ═══════════════════════════════════════════════════════════════════

def get_broker_scatter_data(
    broker: str,
    broker_ts: dict[str, pd.DataFrame],
    price_df: pd.DataFrame,
    lag: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Extract (ΔNP, future_return) pairs for a single broker at a given lag.

    Used to draw scatter plots in the smart-money dashboard tab.

    Parameters
    ----------
    broker : str
        Broker name.
    broker_ts : dict[str, pd.DataFrame]
        Output of calc_broker_momentum_timeseries().
    price_df : pd.DataFrame
        Daily price data with 'date' and 'close' columns.
    lag : int
        Forward-return lag in days.

    Returns
    -------
    pd.DataFrame or None
        Columns: [date, delta_np, net_pos, close, future_return].
        None if insufficient data.
    """
    if broker not in broker_ts:
        return None

    df = broker_ts[broker].sort_values("date").reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    price = price_df.copy()
    if "date" in price.columns:
        price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.sort_values("date").reset_index(drop=True)

    merged = pd.merge(
        df[["date_dt", "delta_np", "net_pos"]],
        price[["date", "close"]],
        left_on="date_dt", right_on="date", how="inner",
    )

    if len(merged) < 3:
        return None

    merged["future_return"] = (
        merged["close"].shift(-lag) / merged["close"] - 1
    )
    merged = merged.dropna(subset=["delta_np", "future_return"])

    if len(merged) < 3:
        return None

    return merged[["date", "delta_np", "net_pos", "close", "future_return"]].reset_index(drop=True)


def compute_structure_alerts_for_all(
    historical_by_symbol: dict,
    broker_ts_by_symbol: dict,
    defections_by_symbol: dict,
) -> list[dict]:
    """
    Compute structure alerts for all symbols by comparing latest vs previous day.

    Returns list of alert dicts, sorted by severity then symbol.
    """
    all_alerts = []
    for symbol in historical_by_symbol:
        hist = historical_by_symbol.get(symbol, {})
        if not hist or len(hist) < 2:
            continue

        # Latest score (all data)
        cr_df = calc_cr_timeseries(hist)
        cr_df_t = calc_cr_trend(cr_df)
        turnover_df = calc_turnover_ratio(hist)
        entrants_df = detect_new_entrants(hist)
        div_df = calc_divergence_index(cr_df)
        defs = defections_by_symbol.get(symbol, [])
        stb = calc_rank_stability(cr_df)

        latest_score = calc_structure_score(
            cr_trend_df=cr_df_t, turnover_df=turnover_df,
            entrants_df=entrants_df, divergence_df=div_df,
            defections=defs, stability_df=stb, symbol=symbol,
        )

        # Previous score (exclude latest day)
        dates = sorted(hist.keys())
        if len(dates) >= 2:
            prev_hist = {d: hist[d] for d in dates[:-1]}
            prev_cr = calc_cr_timeseries(prev_hist)
            prev_cr_t = calc_cr_trend(prev_cr)
            prev_tov = calc_turnover_ratio(prev_hist)
            prev_ent = detect_new_entrants(prev_hist)
            prev_div = calc_divergence_index(prev_cr)
            prev_stb = calc_rank_stability(prev_cr)
            # Filter defections to previous day range
            prev_dates = sorted(prev_hist.keys())
            prev_defs = [d for d in defs if d.get("date", "") <= prev_dates[-1]] if prev_dates else defs

            prev_score = calc_structure_score(
                cr_trend_df=prev_cr_t, turnover_df=prev_tov,
                entrants_df=prev_ent, divergence_df=prev_div,
                defections=prev_defs, stability_df=prev_stb, symbol=symbol,
            )
        else:
            prev_score = None

        alerts = detect_structure_alerts(latest_score, prev_score)
        all_alerts.extend(alerts)

    # Sort: high severity first, then by type priority
    sev_order = {"high": 0, "medium": 1}
    all_alerts.sort(key=lambda a: (sev_order.get(a.get("severity", "low"), 9), a["symbol"]))
    return all_alerts
