# 🔭 Commodity Structure Radar

**龙虎榜微观结构松动雷达** — Institutional Structure Break Radar

A trading decision-support system that finds high-probability reversal signals by detecting **resonance** between institutional seat momentum (from daily DCE position rankings) and false breakout price action (Stop Runs) at key liquidity levels on 30M/1H timeframes.

> **Not** a full auto-trading system. **Not** trend-following or grid-trading.  
> This is a **radar** — it scans for the specific moment when "big money" traps retail traders at key levels.

---

## Trading Philosophy

### Core Principles

1. **Anti-Intuitive Liquidity Mindset**  
   We do NOT analyze bid/ask depth or order book accumulation. Market makers manipulate those. Instead, we focus purely on **price action at key structural levels** — previous day high/low and volume POC.

2. **Seat Momentum = Daily Bias**  
   After market close, we fetch DCE institutional position rankings (龙虎榜) via AkShare. We compute:
   - **Concentration Ratio (CR)**: Are the top 3 institutions dominating one side?
   - **Net Position Momentum (ΔNP)**: Are institutions adding longs or shorts?
   
   This becomes the next day's **bias** — our "battle plan."

3. **Stop Run Engine**  
   We monitor for price "piercing" a key level and then **failing to hold beyond it** within a defined window. This is the classic liquidity sweep / stop run pattern where:
   - Price runs stops above resistance (trapping breakout longs) → reverses down
   - Price runs stops below support (trapping breakdown shorts) → reverses up

4. **Resonance Gate**  
   An alert fires ONLY when the stop-run direction aligns with the daily institutional bias. A bearish stop-run at resistance confirms a bullish bias (traps occurred). A bullish stop-run at support confirms a bearish bias.

---

## Monitored Symbols

| Code | Name | Exchange | Default Contract |
|------|------|----------|-----------------|
| `i` | 铁矿石 (Iron Ore) | DCE | i2509 |
| `rb` | 螺纹钢 (Rebar) | DCE | rb2510 |
| `j` | 焦炭 (Coke) | DCE | j2509 |

Update contracts in `src/config.py` → `MONITORED_SYMBOLS` before each trading cycle.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `akshare`, `tqsdk`, `streamlit`, `plotly`, `pandas`, `pyarrow`, `numpy`

> **Note:** `tqsdk` requires a TqSdk account for real-time data. Anonymous mode is supported (delayed data).

---

## Usage

### Step 1: Fetch Institutional Data (After Market Close)

Run after 15:00 CST when DCE publishes daily position rankings:

```bash
python src/fetcher_akshare.py --date 20260710
```

This will:
1. Fetch DCE position rankings from AkShare
2. Clean and store data to `data/raw/{symbol}_{date}.parquet`
3. Compute Concentration Ratio, ΔNP momentum, and Daily Bias
4. Save `data/bias/daily_bias.json`

### Step 2: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard shows:
- **Sidebar**: Bias summary, CR cards, ΔNP heatmap, key levels
- **K-line Chart**: Plotly candlestick with key level overlays and stop-run markers
- **Alert Feed**: Real-time signals when stop-runs resonate with institutional bias

### Step 3: Interpret Alerts

| Alert | Meaning | Action |
|-------|---------|--------|
| ▲ LONG Stop Run @ prev_low | Price swept below support, rejected up | Bullish entry opportunity |
| ▼ SHORT Stop Run @ prev_high | Price swept above resistance, rejected down | Bearish entry opportunity |
| ✓ Bias共振 | Stop-run direction confirms daily bias | Higher conviction signal |
| ✗ No bias | Stop-run detected but no institutional alignment | Lower conviction |

---

## Project Structure

```
commodity-structure-radar/
├── data/
│   ├── raw/                 # Parquet: {symbol}_{YYYYMMDD}.parquet
│   └── bias/                # daily_bias.json
├── src/
│   ├── __init__.py
│   ├── config.py            # All constants, thresholds, symbols
│   ├── indicators.py        # Pure computation: CR, ΔNP, key levels, stop-run detection
│   ├── fetcher_akshare.py   # AkShare data pipeline
│   └── engine_tqsdk.py      # Real-time TqSdk K-line monitor
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Documentation

📖 **[指标说明与使用手册](docs/indicators-guide.md)** — 所有面板的详细原理解释、指标公式、使用方法和实战场景。

---

## Configuration

All tunable parameters are in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MONITORED_SYMBOLS` | i, rb, j | Symbols with contract codes |
| `TOP_N_CONCENTRATION` | 3 | Top-N brokers for CR |
| `STOPRUN_CONFIRM_WINDOW` | 2 | Max candles for rejection confirmation |
| `DEFAULT_WICK_BODY_RATIO` | 0.6 | Min rejection ratio for valid signal |
| `CR_HIGH_THRESHOLD` | 0.40 | CR above this = concentrated |
| `CR_LOW_THRESHOLD` | 0.20 | CR below this = fragmented (skip) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No bias data available" | Run `python src/fetcher_akshare.py` first |
| TqSdk disconnected | Check internet; TqSdk auto-reconnects |
| Empty AkShare response | It's a weekend/holiday — no data published |
| Wrong contract data | Update `MONITORED_SYMBOLS` contract codes in `config.py` |
| Import errors | `pip install -r requirements.txt` |

---

## License

Internal use. Not for redistribution.
