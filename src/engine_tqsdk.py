"""
engine_tqsdk.py — Real-time K-line monitoring & Stop-Run detection engine.

Runs TqSdk's async event loop in a background thread. Subscribes to 30M/1H
K-lines for the ferrous trio, monitors key levels, and fires alerts when
a stop-run pattern is confirmed AND the direction resonates with daily bias.

Architecture:
    TqSdkEngine (thread runner)
      └─ KeyLevelMonitor (per symbol, tracks price vs levels)
           └─ StopRunDetector (FSM per level, detects false breakouts)
                └─ AlertManager (dedup, persist, expose to UI)
"""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.config import (
    ALERT_DEDUP_WINDOW_MINUTES,
    ALERT_LOG_PATH,
    BIAS_DATA_DIR,
    BIAS_FILE_NAME,
    DEFAULT_WICK_BODY_RATIO,
    KEY_LEVEL_PIERCE_TICKS,
    MONITORED_SYMBOLS,
    STOPRUN_CONFIRM_WINDOW,
    TQSDK_KLINE_1H,
    TQSDK_KLINE_30M,
    TQSDK_QUOTE_TIMEOUT,
    TQSDK_RECONNECT_INTERVAL,
    TQSDK_PASSWORD,
    TQSDK_USERNAME,
)
from src.indicators import detect_stop_run

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. KeyLevelMonitor — tracks price proximity to key levels
# ═══════════════════════════════════════════════════════════════════

class KeyLevelMonitor:
    """
    Tracks real-time price relative to a set of key levels.

    Detects "probe" (near level) and "pierce" (crossed level) events
    that feed into the StopRunDetector state machine.
    """

    def __init__(self, symbol: str, key_levels: dict):
        """
        Parameters
        ----------
        symbol : str
            Symbol code.
        key_levels : dict
            {"prev_high": 792.5, "prev_low": 768.0, "poc": 782.0, ...}
        """
        self.symbol = symbol
        self.key_levels = key_levels
        self.last_price: Optional[float] = None
        # Track whether we were above/below each level on previous tick
        # {level_name: "above"|"below"}
        self._previous_side: dict[str, str] = {}

    def update_price(self, price: float, timestamp: datetime) -> list[dict]:
        """
        Check latest price against all tracked levels.

        Returns a list of events. Each event is:
        {
            "type": "probe_above" | "probe_below" | "pierce_above" | "pierce_below",
            "level_name": str,
            "level_price": float,
            "price": float,
            "timestamp": datetime,
        }
        """
        events = []

        for level_name, level_price in self.key_levels.items():
            if level_price is None:
                continue

            current_side = "above" if price > level_price else "below"
            prev_side = self._previous_side.get(level_name)

            # Detect pierce (crossing the level)
            if prev_side is not None and prev_side != current_side:
                if current_side == "above":
                    events.append({
                        "type": "pierce_above",
                        "level_name": level_name,
                        "level_price": level_price,
                        "price": price,
                        "timestamp": timestamp,
                    })
                else:
                    events.append({
                        "type": "pierce_below",
                        "level_name": level_name,
                        "level_price": level_price,
                        "price": price,
                        "timestamp": timestamp,
                    })

            # Detect probe (within proximity)
            proximity = KEY_LEVEL_PIERCE_TICKS * _get_tick_size(self.symbol)
            if abs(price - level_price) <= proximity:
                events.append({
                    "type": f"probe_{current_side}",
                    "level_name": level_name,
                    "level_price": level_price,
                    "price": price,
                    "timestamp": timestamp,
                })

            self._previous_side[level_name] = current_side

        self.last_price = price
        return events


def _get_tick_size(symbol: str) -> float:
    """Return minimum tick size for a symbol."""
    tick_sizes = {
        # 黑色系
        "i": 0.5,   "rb": 1.0,  "j": 0.5,
        "hc": 1.0,  "jm": 0.5,  "fg": 1.0,
        "sa": 1.0,  "sm": 2.0,  "si": 2.0,
        # 油脂
        "y": 2.0,   "p": 2.0,   "oi": 1.0,
        # 农产品
        "m": 1.0,   "rm": 1.0,  "a": 1.0,   "c": 1.0,
        "b": 1.0,   "cs": 1.0,  "jd": 1.0,  "lh": 5.0,
        # 软商品
        "sr": 1.0,  "cf": 5.0,  "ap": 1.0,
        "ru": 5.0,  "nr": 5.0,  "sp": 2.0,  "ur": 1.0,
        # 化工
        "ma": 1.0,  "ta": 2.0,  "eg": 1.0,  "eb": 1.0,
        "pp": 1.0,  "l": 1.0,   "v": 1.0,
        # 能源
        "sc": 0.1,  "fu": 1.0,  "pg": 1.0,
    }
    return tick_sizes.get(symbol, 1.0)


# ═══════════════════════════════════════════════════════════════════
# 2. StopRunDetector — Finite State Machine
# ═══════════════════════════════════════════════════════════════════

class StopRunDetector:
    """
    State machine tracking the lifecycle of one (symbol, level) pair.

    States: IDLE → PROBING → CONFIRMED | REJECTED → (reset to IDLE)

    The detector is created with an expected direction based on daily bias:
    - bias="bullish" → create detector for prev_high, direction="short"
      (looking for bearish stop-runs at resistance — price runs stops above,
       then reverses down = bullish bias confirmed by trap)
    - bias="bearish" → create detector for prev_low, direction="long"
      (looking for bullish stop-runs at support — price runs stops below,
       then reverses up = bearish bias confirmed by trap)
    """

    def __init__(
        self,
        symbol: str,
        level_name: str,
        level_price: float,
        direction: str,
        wick_ratio: float = DEFAULT_WICK_BODY_RATIO,
        max_candles: int = STOPRUN_CONFIRM_WINDOW,
    ):
        self.symbol = symbol
        self.level_name = level_name
        self.level_price = level_price
        self.direction = direction  # "long" or "short"
        self.wick_ratio = wick_ratio
        self.max_candles = max_candles

        self.state = "IDLE"
        self.pierce_time: Optional[datetime] = None
        self.pierce_price: Optional[float] = None
        self.pierce_candle_ts = None
        self.candles_since_pierce = 0

    def feed_candle(self, candle: pd.Series) -> Optional[dict]:
        """
        Process a completed candle.

        Returns an alert dict if state transitions to CONFIRMED, else None.
        """
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        candle_ts = candle.get("datetime")
        candle_range = h - l

        if candle_range <= 0:
            return None

        if self.state == "IDLE":
            # Check for pierce
            pierced = False
            pierce_price = 0.0
            if self.direction == "long":
                if l < self.level_price:
                    pierced = True
                    pierce_price = l
            else:  # "short"
                if h > self.level_price:
                    pierced = True
                    pierce_price = h

            if pierced:
                self.state = "PROBING"
                self.pierce_time = candle_ts
                self.pierce_price = pierce_price
                self.pierce_candle_ts = candle_ts
                self.candles_since_pierce = 0

            return None

        elif self.state == "PROBING":
            self.candles_since_pierce += 1

            # Check for rejection
            if self.direction == "long":
                # Bullish stop-run: price pierced BELOW support, rejects back above.
                # Rejection ratio = (close - low) / range — captures full upward
                # recovery (both body and lower wick) from the probe low.
                rejection_ratio = (c - l) / candle_range
                closed_above = c > self.level_price

                if closed_above and rejection_ratio >= self.wick_ratio:
                    alert = self._build_alert(candle_ts, c, rejection_ratio)
                    self.reset()
                    return alert

            else:  # "short"
                # Bearish stop-run: price pierced ABOVE resistance, rejects back below.
                # Rejection ratio = (high - close) / range — captures full downward
                # rejection (both body and upper wick) from the probe high.
                rejection_ratio = (h - c) / candle_range
                closed_below = c < self.level_price

                if closed_below and rejection_ratio >= self.wick_ratio:
                    alert = self._build_alert(candle_ts, c, rejection_ratio)
                    self.reset()
                    return alert

            # Window exceeded → reject
            if self.candles_since_pierce >= self.max_candles:
                self.reset()

            return None

        return None

    def _build_alert(self, reject_time, close_price, wick_r) -> dict:
        """Construct alert dict for a confirmed stop-run."""
        severity = "high" if wick_r >= 0.75 else "medium"
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "level": self.level_price,
            "level_name": self.level_name,
            "pierce_time": str(self.pierce_time) if self.pierce_time else "",
            "reject_time": str(reject_time) if reject_time else "",
            "pierce_price": self.pierce_price,
            "close_price": close_price,
            "wick_ratio_observed": round(wick_r, 3),
            "candle_count": self.candles_since_pierce,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

    def reset(self):
        """Return to IDLE. Called after alert is fired or window expires."""
        self.state = "IDLE"
        self.pierce_time = None
        self.pierce_price = None
        self.pierce_candle_ts = None
        self.candles_since_pierce = 0

    def __repr__(self):
        return (
            f"StopRunDetector({self.symbol}/{self.level_name} "
            f"@{self.level_price} dir={self.direction} state={self.state})"
        )


# ═══════════════════════════════════════════════════════════════════
# 3. AlertManager — Dedup, persistence, query
# ═══════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Thread-safe alert registry with deduplication and persistence.

    Dedup rule: no two alerts from the same (symbol, level_name) within
    ALERT_DEDUP_WINDOW_MINUTES (default 120 min).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._alerts: list[dict] = []
        self._acknowledged: set = set()
        # {(symbol, level): time}
        self._last_alert_time: dict[tuple, datetime] = {}

    def push(self, alert: dict) -> bool:
        """
        Insert an alert if it passes the dedup check.

        Returns True if added, False if suppressed.
        """
        with self._lock:
            key = (alert["symbol"], alert["level_name"], alert["direction"])
            now = datetime.now()

            # Dedup check
            if key in self._last_alert_time:
                elapsed = (
                    now - self._last_alert_time[key]).total_seconds() / 60.0
                if elapsed < ALERT_DEDUP_WINDOW_MINUTES:
                    logger.debug(
                        f"Alert suppressed (dedup): {key} — "
                        f"last alert was {elapsed:.0f} min ago"
                    )
                    return False

            # Assign unique ID
            alert_id = f"{alert['symbol']}_{alert['level_name']}_{alert['direction']}_{int(now.timestamp())}"
            alert["id"] = alert_id
            alert["acknowledged"] = False

            self._alerts.append(alert)
            self._last_alert_time[key] = now

            # Trim old alerts if over limit
            from src.config import ALERT_MAX_ACTIVE
            if len(self._alerts) > ALERT_MAX_ACTIVE:
                # Remove acknowledged first, then oldest
                unacked = [a for a in self._alerts if not a.get(
                    "acknowledged", False)]
                acked = [a for a in self._alerts if a.get(
                    "acknowledged", False)]
                if acked:
                    acked.pop(0)
                    self._alerts = unacked + acked
                else:
                    self._alerts = self._alerts[-ALERT_MAX_ACTIVE:]

            # Persist to log
            self._log_alert(alert)

            logger.info(
                f"⚠ ALERT: {alert['symbol']} {alert['direction']} stop-run "
                f"@ {alert['level']} ({alert['level_name']}) "
                f"wick={alert['wick_ratio_observed']:.1%} severity={alert['severity']}"
            )
            return True

    def _log_alert(self, alert: dict) -> None:
        """Append alert to JSONL log file."""
        try:
            os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
            log_entry = alert.copy()
            with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    log_entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    def get_active(self, symbol: Optional[str] = None) -> list[dict]:
        """Return unacknowledged alerts, optionally filtered by symbol."""
        with self._lock:
            alerts = [a for a in self._alerts if not a.get(
                "acknowledged", False)]
            if symbol:
                alerts = [a for a in alerts if a["symbol"] == symbol]
            return sorted(alerts, key=lambda a: a.get("timestamp", ""), reverse=True)

    def get_all(self, symbol: Optional[str] = None) -> list[dict]:
        """Return all alerts (including acknowledged)."""
        with self._lock:
            alerts = list(self._alerts)
            if symbol:
                alerts = [a for a in alerts if a["symbol"] == symbol]
            return sorted(alerts, key=lambda a: a.get("timestamp", ""), reverse=True)

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged. Returns True if found."""
        with self._lock:
            for alert in self._alerts:
                if alert.get("id") == alert_id:
                    alert["acknowledged"] = True
                    return True
            return False

    @property
    def alert_count(self) -> int:
        with self._lock:
            return len([a for a in self._alerts if not a.get("acknowledged", False)])


# ═══════════════════════════════════════════════════════════════════
# 4. TqSdkEngine — Main async runner (runs in background thread)
# ═══════════════════════════════════════════════════════════════════

class TqSdkEngine:
    """
    Wraps TqSdk TqApi in a background thread for use with Streamlit.

    Lifecycle:
    1. __init__: create TqApi, subscribe to K-lines, create detectors.
    2. start(): launch background thread that runs the TqSdk event loop.
    3. On each K-line update, feed completed candles to StopRunDetectors.
    4. stop(): signal thread to exit, close TqApi.

    Streamlit Integration:
        The engine runs in a daemon thread. The main Streamlit thread
        reads from:
        - engine.kline_cache: {symbol: {tf: pd.DataFrame}} (latest candles)
        - engine.alert_manager: AlertManager instance
        - engine.is_connected: bool
    """

    def __init__(self):
        self.symbols = list(MONITORED_SYMBOLS.keys())
        self.api = None
        self.is_running = False
        self.is_connected = False
        self._thread: Optional[threading.Thread] = None

        # Thread-safe caches for Streamlit to read
        self._cache_lock = threading.Lock()
        self.kline_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self.alert_manager = AlertManager()

        # Per-symbol detectors
        self._detectors: dict[str, list[StopRunDetector]] = {}

        # Track processed candle timestamps to avoid duplicates
        self._processed_candles: set = set()

        # Daily bias (loaded before start)
        self._bias: dict[str, dict] = {}
        self._key_levels: dict[str, dict] = {}

        # Reconnection tracking
        self._disconnect_time: Optional[datetime] = None

    # ── Bias loading ──

    def load_bias(self) -> bool:
        """
        Load latest daily_bias.json and create detectors.
        Returns True if bias was loaded successfully.
        """
        bias_path = os.path.join(BIAS_DATA_DIR, BIAS_FILE_NAME)
        if not os.path.exists(bias_path):
            logger.warning(f"Bias file not found: {bias_path}. "
                           f"Run fetcher_akshare.py first.")
            return False

        try:
            with open(bias_path, "r", encoding="utf-8") as f:
                all_bias = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load bias: {e}")
            return False

        if not all_bias:
            logger.warning("Bias file is empty.")
            return False

        # 按实际日期排序获取最新（避免字符串排序导致 "2026-710" 排在 "2026-07-10" 后面）
        valid_dates = []
        for d in all_bias:
            try:
                parts = d.split("-")
                if len(parts) == 3:
                    valid_dates.append((datetime.strptime(d, "%Y-%m-%d"), d))
                elif len(parts) == 2 and len(parts[1]) >= 4:
                    clean = f"{parts[0]}-{parts[1][:2]}-{parts[1][2:]}"
                    valid_dates.append((datetime.strptime(clean, "%Y-%m-%d"), d))
            except (ValueError, IndexError):
                continue
        if not valid_dates:
            logger.warning("No valid dates in bias file.")
            return False
        latest_date = max(valid_dates, key=lambda x: x[0])[1]
        latest = all_bias[latest_date]
        self._bias = latest

        logger.info(f"Loaded bias for {latest_date}: {list(latest.keys())}")

        # Create detectors based on bias direction
        self._detectors = {}
        for symbol in self.symbols:
            symbol_bias = latest.get(symbol, {})
            bias_direction = symbol_bias.get(
                "bias", "skip") if symbol_bias else "skip"
            self._detectors[symbol] = self._create_detectors_for_symbol(
                symbol, bias_direction
            )
            logger.info(
                f"  {symbol}: bias={bias_direction}, "
                f"{len(self._detectors[symbol])} detectors"
            )

        return True

    def _create_detectors_for_symbol(
        self, symbol: str, bias_direction: str
    ) -> list[StopRunDetector]:
        """
        Create StopRunDetectors based on daily bias direction.

        Mapping:
        - BULLISH → monitor prev_high for bearish (short) stop-runs
          (expecting price to sweep above resistance and reverse down)
        - BEARISH → monitor prev_low for bullish (long) stop-runs
          (expecting price to sweep below support and reverse up)
        - NEUTRAL → monitor both levels, both directions
          (higher wick threshold, lower conviction)
        - SKIP → no detectors (no edge)
        """
        detectors = []

        # Use default key levels; these will be updated as candles flow in
        base_levels = {
            "prev_high": None,
            "prev_low": None,
            "poc": None,
        }

        if bias_direction == "bullish":
            # Focus: bearish stop-runs at resistance levels
            detectors.append(StopRunDetector(
                symbol, "prev_high", 0.0, "short",
                wick_ratio=DEFAULT_WICK_BODY_RATIO,
            ))
            # Also monitor POC for stop-runs
            detectors.append(StopRunDetector(
                symbol, "poc", 0.0, "short",
                wick_ratio=DEFAULT_WICK_BODY_RATIO,
            ))

        elif bias_direction == "bearish":
            # Focus: bullish stop-runs at support levels
            detectors.append(StopRunDetector(
                symbol, "prev_low", 0.0, "long",
                wick_ratio=DEFAULT_WICK_BODY_RATIO,
            ))
            detectors.append(StopRunDetector(
                symbol, "poc", 0.0, "long",
                wick_ratio=DEFAULT_WICK_BODY_RATIO,
            ))

        elif bias_direction == "neutral":
            # Monitor both levels, both directions, higher threshold
            higher_wick = min(DEFAULT_WICK_BODY_RATIO + 0.1, 0.8)
            detectors.append(StopRunDetector(
                symbol, "prev_high", 0.0, "short", wick_ratio=higher_wick,
            ))
            detectors.append(StopRunDetector(
                symbol, "prev_low", 0.0, "long", wick_ratio=higher_wick,
            ))

        else:  # skip
            logger.info(f"  {symbol}: bias=skip — no detectors created")

        return detectors

    # ── Background thread ──

    def start(self) -> None:
        """Launch the TqSdk event loop in a background daemon thread."""
        if self.is_running:
            logger.warning("Engine already running.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("TqSdkEngine started in background thread.")

    def _run_loop(self) -> None:
        """Main TqSdk event loop. Runs in background thread."""
        try:
            from tqsdk import TqApi, TqAuth
        except ImportError:
            logger.error("tqsdk not installed. Run: pip install tqsdk")
            self.is_connected = False
            self.is_running = False
            return

        # Connect
        try:
            if TQSDK_USERNAME and TQSDK_PASSWORD:
                self.api = TqApi(auth=TqAuth(TQSDK_USERNAME, TQSDK_PASSWORD))
            else:
                self.api = TqApi()
            self.is_connected = True
            logger.info("TqSdk connected (anonymous mode)")
        except Exception as e:
            logger.error(f"TqSdk connection failed: {e}")
            self.is_connected = False
            self.is_running = False
            return

        # Subscribe to K-lines
        kline_series = {}
        for symbol in self.symbols:
            contract = MONITORED_SYMBOLS[symbol]["contract"]
            exchange = MONITORED_SYMBOLS[symbol]["exchange"]

            # Build TqSdk instrument code (must be lowercase)
            contract_lower = contract.lower()
            code_30m = f"{exchange}.{contract_lower}"
            code_1h = f"{exchange}.{contract_lower}"

            try:
                kline_series[(symbol, "30M")] = self.api.get_kline_serial(
                    code_30m, TQSDK_KLINE_30M
                )
                kline_series[(symbol, "1H")] = self.api.get_kline_serial(
                    code_1h, TQSDK_KLINE_1H
                )
                logger.info(f"Subscribed: {symbol} ({contract}) 30M + 1H")
            except Exception as e:
                logger.error(f"Failed to subscribe {symbol}: {e}")

        # Event loop
        while self.is_running:
            try:
                self.api.wait_update(timeout=TQSDK_QUOTE_TIMEOUT)
                self._process_updates(kline_series)
                self._disconnect_time = None  # clear stale marker
            except Exception as e:
                logger.error(f"TqApi update error: {e}")
                if not self.is_running:
                    break
                if self._disconnect_time is None:
                    self._disconnect_time = datetime.now()
                time.sleep(TQSDK_RECONNECT_INTERVAL)

        # Cleanup
        if self.api:
            try:
                self.api.close()
            except Exception:
                pass
        self.is_connected = False
        logger.info("TqSdkEngine stopped.")

    def _process_updates(self, kline_series: dict) -> None:
        """
        Check all K-line series for newly completed candles.
        Feed completed candles into StopRunDetectors.
        Update kline_cache for UI consumption.
        """
        for (symbol, tf), series_df in kline_series.items():
            if series_df is None or series_df.empty or len(series_df) < 2:
                continue

            # The last completed candle is index -2
            # (index -1 is the currently forming candle)
            last_complete = series_df.iloc[-2]
            candle_ts = last_complete.get("datetime")

            # Dedup: skip already-processed candles
            candle_key = f"{symbol}_{tf}_{candle_ts}"
            if candle_key in self._processed_candles:
                continue
            self._processed_candles.add(candle_key)

            # Trim processed set to avoid unbounded growth
            if len(self._processed_candles) > 10000:
                # Keep only the most recent 5000
                recent = sorted(self._processed_candles)[-5000:]
                self._processed_candles = set(recent)

            # Update key level prices on detectors from latest data
            self._update_detector_levels(symbol, series_df)

            # Feed candle to all detectors for this symbol
            for detector in self._detectors.get(symbol, []):
                if detector.level_price <= 0:
                    continue  # skip if level hasn't been set yet

                alert = detector.feed_candle(last_complete)
                if alert:
                    alert["symbol"] = symbol
                    alert["timeframe"] = tf
                    self.alert_manager.push(alert)

            # Update cache for UI
            with self._cache_lock:
                if symbol not in self.kline_cache:
                    self.kline_cache[symbol] = {}
                self.kline_cache[symbol][tf] = series_df.tail(200).copy()

    def _update_detector_levels(self, symbol: str, series_df: pd.DataFrame) -> None:
        """
        Update detector level prices from the latest K-line data.

        The levels (prev_high, prev_low, poc) are computed dynamically
        from the K-line DataFrame using indicators.calc_key_levels.
        """
        try:
            from src.indicators import calc_key_levels
        except ImportError:
            return

        # Only recompute periodically (every ~10 updates per symbol)
        levels = calc_key_levels(series_df)
        if not levels:
            return

        self._key_levels[symbol] = levels

        # Update detector prices
        for detector in self._detectors.get(symbol, []):
            level_price = levels.get(detector.level_name)
            if level_price is not None and level_price > 0:
                detector.level_price = level_price

    # ── Public API for Streamlit ──

    def stop(self) -> None:
        """Signal the engine to stop. Blocks until thread exits."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("TqSdkEngine stopped cleanly.")

    def get_kline(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get latest K-line DataFrame for a symbol+timeframe."""
        with self._cache_lock:
            cache = self.kline_cache.get(symbol, {})
            df = cache.get(timeframe, pd.DataFrame())
            return df.copy() if not df.empty else pd.DataFrame()

    def get_key_levels(self, symbol: str) -> dict:
        """Get current key levels for a symbol."""
        return self._key_levels.get(symbol, {})

    def get_bias(self, symbol: str) -> dict:
        """Get daily bias for a symbol."""
        return self._bias.get(symbol, {})

    @property
    def is_stale(self) -> bool:
        """True if disconnected for > 5 minutes."""
        if self._disconnect_time is None:
            return False
        return (datetime.now() - self._disconnect_time).total_seconds() > 300
