"""Quick integration test — verifies all modules work together."""
import pandas as pd
from datetime import datetime, timedelta

from src.config import MONITORED_SYMBOLS, STOPRUN_CONFIRM_WINDOW
from src.indicators import (
    calc_concentration_ratio,
    calc_net_position_momentum,
    detect_stop_run,
    calc_daily_bias,
)
from src.engine_tqsdk import StopRunDetector, AlertManager

print("=== INTEGRATION CHECK ===")
print(f"Symbols: {list(MONITORED_SYMBOLS.keys())}")
print(f"SR window: {STOPRUN_CONFIRM_WINDOW} candles")

now = datetime.now()

# Test 1: Bearish stop-run detection
print("\n--- Test 1: Bearish Stop Run ---")
candles = pd.DataFrame([
    {"open": 100, "high": 101, "low": 99, "close": 100.5,
        "datetime": now - timedelta(hours=4)},
    {"open": 100.5, "high": 106, "low": 100, "close": 105,
        "datetime": now - timedelta(hours=3)},
    {"open": 105, "high": 106, "low": 98, "close": 99,
        "datetime": now - timedelta(hours=2)},
])
is_sr, alert = detect_stop_run(
    candles, level=102, direction="short", symbol="test")
print(f"  Triggered: {is_sr}")
if alert:
    print(f"  Rejection ratio: {alert['wick_ratio_observed']:.1%}")
    print(f"  Severity: {alert['severity']}")
    print(f"  Candles after pierce: {alert['candle_count']}")
assert is_sr, "FAIL: Bearish stop-run should be detected"
assert alert["severity"] == "high", "FAIL: Should be high severity"
print("  PASSED")

# Test 2: Bullish stop-run detection
print("\n--- Test 2: Bullish Stop Run ---")
candles2 = pd.DataFrame([
    {"open": 100, "high": 101, "low": 99, "close": 100.5,
        "datetime": now - timedelta(hours=4)},
    {"open": 100.5, "high": 101, "low": 94, "close": 95,
        "datetime": now - timedelta(hours=3)},
    {"open": 95, "high": 102, "low": 93, "close": 101,
        "datetime": now - timedelta(hours=2)},
])
is_sr2, alert2 = detect_stop_run(
    candles2, level=98, direction="long", symbol="test")
print(f"  Triggered: {is_sr2}")
if alert2:
    print(f"  Rejection ratio: {alert2['wick_ratio_observed']:.1%}")
    print(f"  Severity: {alert2['severity']}")
assert is_sr2, "FAIL: Bullish stop-run should be detected"
print("  PASSED")

# Test 3: Concentration ratio
print("\n--- Test 3: Concentration Ratio ---")
df = pd.DataFrame({
    "broker": ["A", "B", "C", "D", "E"],
    "long_pos": [1000, 800, 600, 400, 200],
    "short_pos": [900, 700, 500, 600, 300],
})
cr = calc_concentration_ratio(df)
print(f"  Long CR: {cr['cr_long']:.3f}, Short CR: {cr['cr_short']:.3f}")
assert abs(cr["cr_long"] - 0.8) < 0.01, "FAIL: Long CR should be ~0.8"
print("  PASSED")

# Test 4: Net position momentum
print("\n--- Test 4: Delta NP ---")
df_cur = pd.DataFrame({"broker": ["A", "B"], "long_pos": [
                      1200, 500], "short_pos": [800, 600]})
df_prev = pd.DataFrame({"broker": ["A", "B"], "long_pos": [
                       1000, 600], "short_pos": [900, 500]})
dnp = calc_net_position_momentum(df_cur, df_prev)
print(f"  Delta NP: {dnp}")
assert dnp["A"] > 0, "FAIL: Broker A should have positive delta"
print("  PASSED")

# Test 5: Daily bias
print("\n--- Test 5: Daily Bias ---")
bias = calc_daily_bias("test", cr_long=0.45, cr_short=0.18,
                       delta_np={"A": 300, "B": -100}, date="2026-07-11")
print(f"  Bias: {bias['bias']} (conviction: {bias['conviction']:.2f})")
assert bias["bias"] == "bullish", "FAIL: Should be bullish"
print("  PASSED")

# Test 6: AlertManager dedup
print("\n--- Test 6: AlertManager Dedup ---")
am = AlertManager()
a1 = {"symbol": "i", "level_name": "prev_high", "direction": "short",
      "level": 800, "wick_ratio_observed": 0.7, "severity": "high"}
result1 = am.push(a1)
result2 = am.push(a1)  # immediate duplicate
print(f"  First push: {result1}, Duplicate push: {result2}")
assert result1 is True, "FAIL: First alert should be accepted"
assert result2 is False, "FAIL: Duplicate should be suppressed"
active = am.get_active()
assert len(active) == 1, "FAIL: Should have 1 active alert"
am.acknowledge(active[0]["id"])
assert am.alert_count == 0, "FAIL: Should have 0 active after ack"
print("  PASSED")

print("\n=== ALL INTEGRATION TESTS PASSED ===")
