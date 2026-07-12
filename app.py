"""
app.py — 龙虎榜微观结构松动雷达 · Streamlit 看板

Usage:
    streamlit run app.py
"""

import json
import logging
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    BIAS_DATA_DIR,
    BIAS_FILE_NAME,
    MONITORED_SYMBOLS,
    SYMBOL_NAMES,
    UI_CHART_CANDLES,
    UI_REFRESH_INTERVAL_SECONDS,
    STRUCTURE_LOOKBACK_DAYS,
    CR_TREND_WINDOW,
)
from src.indicators import calc_key_levels
from src.fetcher_akshare import load_historical_positions, load_price_data
from src.structural_analysis import (
    calc_cr_timeseries,
    calc_cr_trend,
    detect_structure_change,
    calc_broker_momentum_timeseries,
    calc_rank_stability,
    detect_broker_defections,
    calc_broker_price_correlation,
    identify_smart_money,
    calc_turnover_ratio,
    detect_new_entrants,
    calc_divergence_index,
    calc_concentration_profile_timeseries,
    calc_structure_score,
    calc_all_symbols_scores,
    detect_structure_alerts,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 页面设置 ──
st.set_page_config(
    page_title="结构松动雷达",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
# 会话状态初始化
# ═══════════════════════════════════════════════════════════════════

def init_session_state():
    if "engine" not in st.session_state:
        st.session_state.engine = None
        st.session_state.is_online = False
        st.session_state.bias_data = {}
        st.session_state.selected_symbol = "i"
        st.session_state.timeframe = "30M"
        st.session_state.alerts = []
        st.session_state.alert_ids = set()
        st.session_state.engine_started = False

    if not st.session_state.bias_data:
        st.session_state.bias_data = _load_bias_file()

    if not st.session_state.engine_started:
        _try_start_engine()


def _load_bias_file() -> dict:
    """加载 daily_bias.json，返回最新日期的偏见数据。"""
    bias_path = os.path.join(BIAS_DATA_DIR, BIAS_FILE_NAME)
    if not os.path.exists(bias_path):
        return {}
    try:
        with open(bias_path, "r", encoding="utf-8") as f:
            all_bias = json.load(f)
        if not all_bias:
            return {}
        # 按实际日期排序（而非字符串排序），避免 "2026-710" 之类异常键
        from datetime import datetime
        valid = []
        for d in all_bias:
            try:
                parts = d.split("-")
                if len(parts) == 3:
                    valid.append((datetime.strptime(d, "%Y-%m-%d"), d))
                elif len(parts) == 2 and len(parts[1]) >= 4:
                    clean = f"{parts[0]}-{parts[1][:2]}-{parts[1][2:]}"
                    valid.append((datetime.strptime(clean, "%Y-%m-%d"), d))
            except (ValueError, IndexError):
                continue
        if not valid:
            return {}
        latest_key = max(valid, key=lambda x: x[0])[1]
        return all_bias[latest_key]
    except Exception:
        return {}


def _try_start_engine():
    try:
        from src.engine_tqsdk import TqSdkEngine
        engine = TqSdkEngine()
        if engine.load_bias():
            engine.start()
            st.session_state.engine = engine
            st.session_state.is_online = True
        else:
            st.session_state.engine = None
            st.session_state.is_online = False
    except Exception as e:
        logger.error(f"引擎启动失败: {e}")
        st.session_state.engine = None
        st.session_state.is_online = False
    st.session_state.engine_started = True


# ═══════════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.title("🔭 结构松动雷达")
        st.caption("龙虎榜 · 微观结构 · 流动性猎杀")

        # ── 连接状态 ──
        engine = st.session_state.engine
        is_online = st.session_state.is_online

        if is_online and engine and engine.is_connected:
            stale = engine.is_stale
            if stale:
                st.warning("⚠ 天勤：重连中…")
            else:
                st.success("● 天勤：已连接")
        else:
            st.error("○ 天勤：未连接")
            st.caption("先执行 `python src/fetcher_akshare.py`，再刷新页面。")

        st.divider()

        # ── 品种选择 ──
        symbol = st.selectbox(
            "合约",
            options=list(MONITORED_SYMBOLS.keys()),
            format_func=lambda s: f"{s} — {SYMBOL_NAMES.get(s, s)} "
                                   f"({MONITORED_SYMBOLS[s]['contract']})",
            key="symbol_selector",
        )
        st.session_state.selected_symbol = symbol

        # ── 周期切换 ──
        timeframe = st.radio(
            "K 线周期",
            ["30M", "1H"],
            horizontal=True,
            key="timeframe_radio",
        )
        st.session_state.timeframe = timeframe

        st.divider()

        # ── 每日偏见摘要 ──
        bias = st.session_state.bias_data.get(symbol, {})

        if bias:
            st.subheader("📊 席位偏见")

            bias_direction = bias.get("bias", "skip")
            conviction = bias.get("conviction", 0)

            bias_cn = {"bullish": "看多 ▲", "bearish": "看空 ▼", "neutral": "中性 ◆", "skip": "跳过 ◇"}
            bias_color = {"bullish": "green", "bearish": "red", "neutral": "orange", "skip": "gray"}

            st.markdown(
                f"### :{bias_color.get(bias_direction, 'gray')}"
                f"[{bias_cn.get(bias_direction, '未知')}]"
            )

            st.caption(f"确信度：{conviction:.0%}")
            st.progress(conviction)

            cr_cols = st.columns(2)
            with cr_cols[0]:
                st.metric("多头集中度", f"{bias.get('cr_long', 0):.1%}")
            with cr_cols[1]:
                st.metric("空头集中度", f"{bias.get('cr_short', 0):.1%}")

            aggregate_dnp = bias.get("aggregate_delta_np", 0)
            st.metric("净持仓动量 (ΔNP)", f"{aggregate_dnp:+,}")

            top_bullish = bias.get("top_bullish", [])
            top_bearish = bias.get("top_bearish", [])

            if top_bullish:
                st.caption(f"▲ 加多: {', '.join(top_bullish[:3])}")
            if top_bearish:
                st.caption(f"▼ 加空: {', '.join(top_bearish[:3])}")

            details = bias.get("details", "")
            if details:
                with st.expander("📋 详细逻辑"):
                    st.caption(details)

            st.divider()

            # ── 关键价位 ──
            if engine and is_online and engine.is_connected:
                levels = engine.get_key_levels(symbol)
            else:
                levels = {}

            if levels:
                st.subheader("📍 关键价位")
                level_labels = {
                    "prev_high": ("前日高点", "#FFA726"),
                    "prev_low": ("前日低点", "#FFA726"),
                    "poc": ("成交量密集区 POC", "#42A5F5"),
                    "today_open": ("今日开盘", "#AB47BC"),
                }
                for lname, lprice in levels.items():
                    label, lcolor = level_labels.get(lname, (lname, "#ccc"))
                    if lprice is not None:
                        st.caption(f"▎{label}: **{lprice}**")
        else:
            st.info("暂无偏见数据。\n\n请先运行 `python src/fetcher_akshare.py` 抓取数据。")

        st.divider()

        # ── 操作按钮 ──
        if st.button("🔄 刷新数据", use_container_width=True):
            st.session_state.bias_data = _load_bias_file()
            st.rerun()

        if is_online and engine:
            if st.button("⏹ 停止引擎", use_container_width=True):
                engine.stop()
                st.session_state.is_online = False
                st.session_state.engine_started = False
                st.rerun()


# ═══════════════════════════════════════════════════════════════════
# K 线图 (Plotly)
# ═══════════════════════════════════════════════════════════════════

def render_chart(symbol: str, timeframe: str):
    engine = st.session_state.engine
    is_online = st.session_state.is_online

    if is_online and engine and engine.is_connected:
        df = engine.get_kline(symbol, timeframe)
    else:
        df = pd.DataFrame()

    if df.empty:
        st.info(f"等待 {symbol.upper()} {timeframe} K 线数据…")
        return

    levels = calc_key_levels(df)
    if not levels and engine and is_online:
        levels = engine.get_key_levels(symbol)

    if is_online and engine:
        symbol_alerts = engine.alert_manager.get_active(symbol)
    else:
        symbol_alerts = []

    # ── Plotly 图表 ──
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.get("datetime", df.index),
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name=symbol.upper(),
        increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
        decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
        showlegend=False,
    ))

    # 关键价位水平线
    level_styles = {
        "prev_high": ("前高", "dash", "#FFA726"),
        "prev_low": ("前低", "dash", "#FFA726"),
        "poc": ("POC", "solid", "#42A5F5"),
        "today_open": ("开盘", "dot", "#AB47BC"),
    }

    for level_name, price in levels.items():
        if price is None:
            continue
        label, dash_style, color = level_styles.get(level_name, (level_name, "dash", "#888"))
        fig.add_hline(
            y=price, line_dash=dash_style, line_color=color, line_width=1.2,
            annotation_text=f" {label} ({price})",
            annotation_position="right", annotation_font_size=10,
        )

    # Stop Run 信号标记
    dir_cn = {"long": "多头陷阱", "short": "空头陷阱"}
    for alert in symbol_alerts:
        direction = alert.get("direction", "long")
        marker_color = "#00E676" if direction == "long" else "#FF1744"
        reject_time = alert.get("reject_time", "")

        fig.add_trace(go.Scatter(
            x=[reject_time] if reject_time else [None],
            y=[alert.get("level", 0)],
            mode="markers",
            marker=dict(symbol="x", size=14, color=marker_color,
                        line=dict(width=2, color="white")),
            name=dir_cn.get(direction, direction),
            showlegend=False,
            hovertemplate=(
                f"<b>{dir_cn.get(direction, direction)}</b><br>"
                f"价位: {alert.get('level', '?')}<br>"
                f"影线比: {alert.get('wick_ratio_observed', 0):.1%}<br>"
                f"强度: {alert.get('severity', '?')}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=10, r=20, t=30, b=10),
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=11),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="#333333"),
        yaxis=dict(title=symbol.upper(), gridcolor="#333333", side="right"),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}_{timeframe}")


# ═══════════════════════════════════════════════════════════════════
# 信号推送
# ═══════════════════════════════════════════════════════════════════

def render_alert_feed():
    st.subheader("🚨 信号推送")

    engine = st.session_state.engine
    is_online = st.session_state.is_online

    if is_online and engine:
        all_alerts = engine.alert_manager.get_all()
        active_count = engine.alert_manager.alert_count
    else:
        all_alerts = []
        active_count = 0

    st.caption(f"未读: {active_count} | 总计: {len(all_alerts)}")

    if not all_alerts:
        st.info(
            f"正在监控 **{'、'.join(MONITORED_SYMBOLS.keys())}**"
            f"，暂无信号。\n\n"
            f"当价格行为与席位偏见共振时，信号将在此出现。"
        )
        return

    dir_cn = {"long": "多头陷阱 ▲", "short": "空头陷阱 ▼"}
    severity_cn = {"high": "🔴 高", "medium": "🟡 中"}

    for alert in all_alerts[-30:][::-1]:
        is_acked = alert.get("acknowledged", False)
        alert_symbol = alert.get("symbol", "?")
        direction = alert.get("direction", "?")
        severity = alert.get("severity", "medium")
        wick_r = alert.get("wick_ratio_observed", 0)
        level_name = alert.get("level_name", "?")
        level = alert.get("level", 0)

        level_cn = {"prev_high": "前高", "prev_low": "前低", "poc": "POC"}.get(level_name, level_name)

        with st.container(border=True):
            header_cols = st.columns([2, 3, 2, 1])

            with header_cols[0]:
                st.markdown(f"**{alert_symbol.upper()}** — {SYMBOL_NAMES.get(alert_symbol, '')}")
                st.caption(alert.get("timestamp", "")[:19])

            with header_cols[1]:
                dir_color = "#00E676" if direction == "long" else "#FF1744"
                st.markdown(
                    f"<span style='color:{dir_color};font-size:1.1em'>"
                    f"{dir_cn.get(direction, direction)}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{level_cn} @ {level} | 影线比: {wick_r:.1%} | {alert.get('candle_count', '?')} 根K线")

            with header_cols[2]:
                st.caption(severity_cn.get(severity, severity))
                st.caption(f"穿刺价: {alert.get('pierce_price', '?')} → 收盘: {alert.get('close_price', '?')}")

            with header_cols[3]:
                if not is_acked and is_online and engine:
                    if st.button("✓ 已阅", key=f"ack_{alert.get('id', id(alert))}"):
                        engine.alert_manager.acknowledge(alert["id"])
                        st.rerun()
                elif is_acked:
                    st.caption("✓ 已读")


# ═══════════════════════════════════════════════════════════════════
# 结构演变 — CR 时间序列 + 松动/收紧检测
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def _load_structure_data(symbol: str):
    """Load and compute structural analysis for a symbol. Cached 5 min."""
    hist = load_historical_positions(symbol, lookback_days=STRUCTURE_LOOKBACK_DAYS)
    if not hist:
        return None, None, None, None

    cr_df = calc_cr_timeseries(hist)
    cr_df = calc_cr_trend(cr_df)
    events = detect_structure_change(cr_df)
    stability = calc_rank_stability(cr_df)

    return cr_df, events, stability, hist


@st.cache_data(ttl=600)
def _load_p1_data(symbol: str):
    """Load broker-level P1 analysis data. Cached 10 min."""
    hist = load_historical_positions(symbol, lookback_days=STRUCTURE_LOOKBACK_DAYS)
    if not hist or len(hist) < 2:
        return None, None, None

    broker_ts = calc_broker_momentum_timeseries(hist, top_n=5)
    defections = detect_broker_defections(broker_ts)

    # Try loading price data for correlation
    price_df = load_price_data(symbol)
    corr_df = None
    smart_money = None
    if price_df is not None and not price_df.empty:
        corr_df = calc_broker_price_correlation(broker_ts, price_df)
        if corr_df is not None and not corr_df.empty:
            smart_money = identify_smart_money(corr_df)

    return broker_ts, defections, corr_df, smart_money


@st.cache_data(ttl=300)
def _load_p2_data(symbol: str):
    """Load P2 microstructure indicators. Cached 5 min."""
    hist = load_historical_positions(symbol, lookback_days=STRUCTURE_LOOKBACK_DAYS)
    if not hist or len(hist) < 2:
        return None, None, None, None

    turnover_df = calc_turnover_ratio(hist)
    entrants_df = detect_new_entrants(hist)
    profile_df = calc_concentration_profile_timeseries(hist)

    cr_df = calc_cr_timeseries(hist)
    divergence_df = calc_divergence_index(cr_df)

    return turnover_df, entrants_df, divergence_df, profile_df


@st.cache_data(ttl=600)
def _load_all_symbols_scores():
    """Compute structure scores for ALL monitored symbols. Cached 10 min."""
    scores = []
    for sym in MONITORED_SYMBOLS:
        hist = load_historical_positions(sym, lookback_days=STRUCTURE_LOOKBACK_DAYS)
        if not hist or len(hist) < 2:
            continue
        cr_df = calc_cr_timeseries(hist)
        cr_df = calc_cr_trend(cr_df)
        turnover_df = calc_turnover_ratio(hist)
        entrants_df = detect_new_entrants(hist)
        divergence_df = calc_divergence_index(cr_df)
        broker_ts = calc_broker_momentum_timeseries(hist, top_n=5)
        defections = detect_broker_defections(broker_ts)
        stability_df = calc_rank_stability(cr_df)
        score = calc_structure_score(
            cr_trend_df=cr_df, turnover_df=turnover_df,
            entrants_df=entrants_df, divergence_df=divergence_df,
            defections=defections, stability_df=stability_df, symbol=sym,
        )
        scores.append(score)
    scores.sort(key=lambda s: s["score"])
    return scores


def render_structure_heatmap():
    """Cross-symbol structure score heatmap — all symbols at a glance."""
    scores = _load_all_symbols_scores()
    if not scores:
        st.info("至少需要 2 天数据。")
        return

    import plotly.graph_objects as go

    h = sum(1 for s in scores if s["status"] == "healthy")
    c = sum(1 for s in scores if s["status"] == "caution")
    u = sum(1 for s in scores if s["status"] == "unstable")
    x = sum(1 for s in scores if s["status"] == "critical")

    scols = st.columns(5)
    scols[0].metric("总品种", len(scores))
    scols[1].metric("🟢 健康", h)
    scols[2].metric("🟡 关注", c)
    scols[3].metric("🟠 松动", u)
    scols[4].metric("🔴 危险", x, delta="需关注" if x > 0 else None)

    status_colors = {"healthy": "#00E676", "caution": "#FFEB3B",
                     "unstable": "#FFA726", "critical": "#FF1744"}
    status_cn = {"healthy": "健康", "caution": "关注",
                 "unstable": "松动", "critical": "危险"}

    names = [f"{s['symbol']} {SYMBOL_NAMES.get(s['symbol'], '')}" for s in scores]
    values = [s["score"] for s in scores]
    colors = [status_colors.get(s["status"], "#888") for s in scores]
    custom = [status_cn.get(s["status"], s["status"]) for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h", marker=dict(color=colors),
        text=[f"{v}分" for v in values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>评分: %{x}/100<br>状态: %{customdata}<extra></extra>",
        customdata=custom,
    ))
    for th, lb, lc in [(80, "健康线", "#00E676"), (60, "关注线", "#FFEB3B"), (40, "危险线", "#FF1744")]:
        fig.add_vline(x=th, line_dash="dash", line_color=lc,
                      annotation_text=lb, annotation_position="top", opacity=0.5)
    fig.update_layout(
        template="plotly_dark", height=max(300, len(scores) * 32 + 60),
        margin=dict(l=10, r=60, t=20, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=11),
        xaxis=dict(gridcolor="#333333", title="结构健康度", range=[0, 105]),
        yaxis=dict(gridcolor="#333333", autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, key="heatmap_bar")

    with st.expander("📋 评分明细", expanded=False):
        rows = []
        label_map = {"cr_stability": "CR稳定", "turnover": "换手率",
                     "new_entrants": "新进入者", "divergence": "分歧度",
                     "defections": "叛变", "rank_stability": "排名稳定"}
        for s in scores:
            row = {"品种": f"{s['symbol']} {SYMBOL_NAMES.get(s['symbol'], '')}",
                   "总分": s["score"],
                   "状态": {"healthy": "🟢健康", "caution": "🟡关注",
                            "unstable": "🟠松动", "critical": "🔴危险"}.get(s["status"], ""),}
            for cn, c in s.get("components", {}).items():
                row[label_map.get(cn, cn)] = f"{c['score']}/{c['max']}"
            row["摘要"] = s.get("summary", "")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_panorama(symbol: str):
    """Render P2+P3 microstructure panorama with composite score."""
    turnover_df, entrants_df, divergence_df, profile_df = _load_p2_data(symbol)

    if turnover_df is None or turnover_df.empty:
        st.info("数据不足，至少需要 2 天持仓数据。")
        return

    # ═══ Composite Score Card ═══
    hist = load_historical_positions(symbol, lookback_days=STRUCTURE_LOOKBACK_DAYS)
    cr_df = calc_cr_timeseries(hist)
    cr_df = calc_cr_trend(cr_df)
    broker_ts = calc_broker_momentum_timeseries(hist, top_n=5)
    defections = detect_broker_defections(broker_ts)
    stability_df = calc_rank_stability(cr_df)

    score = calc_structure_score(
        cr_trend_df=cr_df, turnover_df=turnover_df,
        entrants_df=entrants_df, divergence_df=divergence_df,
        defections=defections, stability_df=stability_df, symbol=symbol,
    )

    status_cfg = {
        "healthy": ("🟢 结构健康", "#00E676"),
        "caution": ("🟡 值得关注", "#FFEB3B"),
        "unstable": ("🟠 结构松动", "#FFA726"),
        "critical": ("🔴 结构危险", "#FF1744"),
    }
    status_label, status_color = status_cfg.get(score["status"], ("?", "#888"))

    score_cols = st.columns([2, 1, 3])
    with score_cols[0]:
        st.markdown(
            f"### <span style='color:{status_color}'>{status_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(score["summary"])
    with score_cols[1]:
        st.metric("结构评分", f"{score['score']}/100")
    with score_cols[2]:
        # Mini bar chart of component scores
        comp_names = []
        comp_pct = []
        for cn, c in score["components"].items():
            nm = {"cr_stability": "CR", "turnover": "换手", "new_entrants": "新进",
                  "divergence": "分歧", "defections": "叛变", "rank_stability": "排名"}
            comp_names.append(nm.get(cn, cn))
            comp_pct.append(c["score"] / max(c["max"], 1))
        comp_colors = ["#00E676" if p >= 0.7 else "#FFEB3B" if p >= 0.4 else "#FF1744"
                       for p in comp_pct]
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp_pct, y=comp_names, orientation="h",
            marker=dict(color=comp_colors),
            text=[f"{c['score']}/{c['max']}" for c in score["components"].values()],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", height=120, margin=dict(l=10, r=60, t=5, b=5),
            paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
            font=dict(color="#CCCCCC", size=9),
            xaxis=dict(range=[0, 1.15], showticklabels=False, showgrid=False),
            yaxis=dict(gridcolor="#333333"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="score_components")

    st.divider()

    # ═══ Row 1: Turnover + New Entrant Ratio ═══
    st.caption("**换手率 & 新进入者** — 结构松动的两个前瞻指标")
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        _render_turnover_chart(turnover_df)

    with r1_col2:
        _render_entrants_chart(entrants_df)

    # ═══ Row 2: Divergence + Concentration Profile ═══
    st.caption("**多空博弈 & 集中度剖面** — 从不同粒度看结构演变")
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        _render_divergence_chart(divergence_df)

    with r2_col2:
        _render_profile_chart(profile_df)


def _render_turnover_chart(turnover_df: pd.DataFrame):
    """Turnover rate — line chart with threshold bands."""
    import plotly.graph_objects as go
    from src.config import TURNOVER_HIGH_THRESHOLD, TURNOVER_SPIKE_THRESHOLD

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=turnover_df["date"], y=turnover_df["total_turnover"],
        mode="lines+markers", name="总换手率",
        line=dict(color="#42A5F5", width=2),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(66,165,245,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=turnover_df["date"], y=turnover_df["long_turnover"],
        mode="lines", name="多头换手", line=dict(color="#00E676", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=turnover_df["date"], y=turnover_df["short_turnover"],
        mode="lines", name="空头换手", line=dict(color="#FF1744", width=1, dash="dot"),
    ))

    # Thresholds
    fig.add_hline(y=TURNOVER_HIGH_THRESHOLD, line_dash="dash", line_color="#FFA726",
                  annotation_text="高换手", annotation_position="right")
    fig.add_hline(y=TURNOVER_SPIKE_THRESHOLD, line_dash="dash", line_color="#FF1744",
                  annotation_text="剧烈", annotation_position="right")

    # Mark spike days
    spike_days = turnover_df[turnover_df["is_spike"]]
    if not spike_days.empty:
        fig.add_trace(go.Scatter(
            x=spike_days["date"], y=spike_days["total_turnover"],
            mode="markers", marker=dict(symbol="x", size=12, color="#FF1744"),
            name="剧烈换手", showlegend=False,
            hovertemplate="<b>剧烈换手日</b><br>换手率: %{y:.1%}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=10),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333", tickformat=".0%", title="换手率"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="turnover_chart")


def _render_entrants_chart(entrants_df: pd.DataFrame):
    """New entrant ratio — bar chart."""
    import plotly.graph_objects as go

    if entrants_df is None or entrants_df.empty:
        st.caption("新进入者数据不足")
        return

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=entrants_df["date"], y=entrants_df["new_entrants"],
        name="新进入者数", marker_color="#FFA726",
        hovertemplate="新进: %{y}席<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=entrants_df["date"], y=entrants_df["new_entrant_ratio"],
        mode="lines+markers", name="新进入者占比",
        line=dict(color="#FF1744", width=2),
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate="占比: %{y:.0%}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=10, r=50, t=20, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=10),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333", title="新增席位数"),
        yaxis2=dict(
            title="占比", tickformat=".0%", overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)", range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="entrants_chart")

    # New entrant names for most recent day
    if len(entrants_df) > 0:
        last = entrants_df.iloc[-1]
        if last.get("new_entrants", 0) > 0:
            names = last.get("entrant_names", [])
            st.caption(f"最新: **{', '.join(names[:5])}**" +
                      (f" 等 {len(names)} 个席位" if len(names) > 5 else "") +
                      f" 新进入 Top-20")


def _render_divergence_chart(divergence_df: pd.DataFrame):
    """Divergence index + force balance chart."""
    import plotly.graph_objects as go

    if divergence_df is None or divergence_df.empty:
        st.caption("分歧度数据不足")
        return

    fig = go.Figure()

    # Divergence index as area
    fig.add_trace(go.Scatter(
        x=divergence_df["date"], y=divergence_df["divergence_index"],
        mode="lines", name="分歧度指数",
        line=dict(color="#AB47BC", width=2),
        fill="tozeroy", fillcolor="rgba(171,71,188,0.1)",
    ))

    # Force balance as color-coded bars
    fb = divergence_df["force_balance"]
    colors = ["#00E676" if v > 0 else "#FF1744" for v in fb]
    fig.add_trace(go.Bar(
        x=divergence_df["date"], y=fb,
        name="力量天平", marker_color=colors, opacity=0.6,
        yaxis="y2",
    ))

    # Mark convergence points
    conv = divergence_df[divergence_df["convergence"]]
    if not conv.empty:
        fig.add_trace(go.Scatter(
            x=conv["date"], y=conv["divergence_index"],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#FFEB3B"),
            name="收敛信号", showlegend=False,
            hovertemplate="<b>力量收敛</b><extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=10, r=50, t=20, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=10),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333", title="分歧度 (0=极端倾斜, 1=完全平衡)", range=[0, 1]),
        yaxis2=dict(
            title="多空差", overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="divergence_chart")


def _render_profile_chart(profile_df: pd.DataFrame):
    """Multi-level CR + Gini chart."""
    import plotly.graph_objects as go

    if profile_df is None or profile_df.empty:
        st.caption("集中度剖面数据不足")
        return

    fig = go.Figure()

    # CR at multiple levels — long side
    for n, color, dash in [(1, "#00E676", "dot"), (3, "#00E676", "solid"),
                             (5, "#00E676", "dash"), (10, "#00E676", "dashdot")]:
        col = f"cr_{n}_long"
        if col in profile_df.columns:
            fig.add_trace(go.Scatter(
                x=profile_df["date"], y=profile_df[col],
                mode="lines", name=f"CR{n} 多头",
                line=dict(color=color, width=1.5, dash=dash),
                legendgroup="long", legendgrouptitle_text="多头侧",
            ))

    # CR at multiple levels — short side
    for n, color, dash in [(1, "#FF1744", "dot"), (3, "#FF1744", "solid"),
                             (5, "#FF1744", "dash"), (10, "#FF1744", "dashdot")]:
        col = f"cr_{n}_short"
        if col in profile_df.columns:
            fig.add_trace(go.Scatter(
                x=profile_df["date"], y=profile_df[col],
                mode="lines", name=f"CR{n} 空头",
                line=dict(color=color, width=1.5, dash=dash),
                legendgroup="short", legendgrouptitle_text="空头侧",
            ))

    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=10),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333", tickformat=".0%",
                   title="集中度 (CR1=第一大, CR10=前十大)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=8)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="profile_chart")

    # Gini summary
    st.caption("**Gini 系数** — 0=完全均分, 1=单一垄断")
    gini_cols = st.columns(2)
    if "gini_long" in profile_df.columns and not profile_df.empty:
        gini_l = profile_df["gini_long"].iloc[-1]
        gini_s = profile_df["gini_short"].iloc[-1]
        with gini_cols[0]:
            st.metric("多头 Gini", f"{gini_l:.3f}")
        with gini_cols[1]:
            st.metric("空头 Gini", f"{gini_s:.3f}")


def render_structure_evolution():
    """Render the CR time series chart and structure change signals."""
    symbol = st.session_state.selected_symbol

    cr_df, events, stability, hist = _load_structure_data(symbol)

    if cr_df is None or cr_df.empty or len(cr_df) < 1:
        st.info(f"需要至少 2 个交易日的 `{symbol}` 持仓数据才能分析结构演变。\n\n"
                f"请运行 `python src/fetcher_akshare.py --date YYYYMMDD` 积累更多数据。")
        return

    st.subheader(f"📈 {symbol.upper()} 持仓结构演变")

    # ── Metric row ──
    metric_cols = st.columns(5)

    latest = cr_df.iloc[-1]
    with metric_cols[0]:
        st.metric("CR 多头", f"{latest['cr_long']:.1%}",
                  delta=f"{latest.get('cr_long_slope', 0):+.3f}/day" if len(cr_df) >= CR_TREND_WINDOW else None)
    with metric_cols[1]:
        st.metric("CR 空头", f"{latest['cr_short']:.1%}",
                  delta=f"{latest.get('cr_short_slope', 0):+.3f}/day" if len(cr_df) >= CR_TREND_WINDOW else None)
    with metric_cols[2]:
        cr_diff = latest["cr_diff"]
        st.metric("CR 多空差", f"{cr_diff:+.1%}")
    with metric_cols[3]:
        structure_long = latest.get("structure_long", "stable")
        structure_short = latest.get("structure_short", "stable")
        long_emoji = {"tightening": "🔒", "loosening": "🔓", "stable": "➖"}.get(structure_long, "➖")
        short_emoji = {"tightening": "🔒", "loosening": "🔓", "stable": "➖"}.get(structure_short, "➖")
        st.metric("结构状态", f"多头{structure_long} 空头{structure_short}",
                  delta=f"{long_emoji}多 / {short_emoji}空")
    with metric_cols[4]:
        hhi_long = latest.get("hhi_long", 0)
        hhi_short = latest.get("hhi_short", 0)
        st.metric("HHI 多头/空头", f"{hhi_long:.3f} / {hhi_short:.3f}")

    # ── CR Time Series Chart ──
    broker_ts, defections, corr_df, smart_money = _load_p1_data(symbol)

    chart_tab, panorama_tab, events_tab, broker_tab, defection_tab, smart_tab = st.tabs([
        "📈 CR 趋势图", "📊 全景", "🚨 结构事件", "🔍 席位排名变动",
        "🚨 席位叛变", "💰 聪明钱",
    ])

    with chart_tab:
        _render_cr_chart(cr_df)

    with panorama_tab:
        _render_panorama(symbol)

    with events_tab:
        _render_structure_events(events, cr_df)

    with broker_tab:
        _render_broker_changes(stability, hist, symbol)

    with defection_tab:
        _render_defection_events(defections, symbol)

    with smart_tab:
        _render_smart_money(corr_df, smart_money, symbol)


def _render_cr_chart(cr_df: pd.DataFrame):
    """Plot CR_long, CR_short, and structure bands."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # CR lines
    fig.add_trace(go.Scatter(
        x=cr_df["date"], y=cr_df["cr_long"],
        mode="lines+markers", name="CR 多头",
        line=dict(color="#00E676", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=cr_df["date"], y=cr_df["cr_short"],
        mode="lines+markers", name="CR 空头",
        line=dict(color="#FF1744", width=2),
        marker=dict(size=6),
    ))

    # Threshold reference lines
    from src.config import CR_HIGH_THRESHOLD, CR_LOW_THRESHOLD
    fig.add_hline(y=CR_HIGH_THRESHOLD, line_dash="dash", line_color="#FFA726",
                  annotation_text=f"高集中({CR_HIGH_THRESHOLD:.0%})",
                  annotation_position="right")
    fig.add_hline(y=CR_LOW_THRESHOLD, line_dash="dash", line_color="#888888",
                  annotation_text=f"分散({CR_LOW_THRESHOLD:.0%})",
                  annotation_position="right")

    # Mark loosening/tightening periods if enough data
    if len(cr_df) >= CR_TREND_WINDOW:
        loose_mask = cr_df["structure_long"] == "loosening"
        if loose_mask.any():
            loose_dates = cr_df.loc[loose_mask, "date"]
            for d in loose_dates:
                fig.add_vline(x=d, line_width=0, fillcolor="rgba(255,23,68,0.08)",
                             fill_type="overlay", layer="below")

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=20, t=20, b=10),
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=11),
        xaxis=dict(gridcolor="#333333", title=None),
        yaxis=dict(gridcolor="#333333", title="集中度", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True, key="cr_chart")

    # ── Supplementary: HHI and total position sub-charts ──
    sub_col1, sub_col2 = st.columns(2)

    with sub_col1:
        fig_hhi = go.Figure()
        fig_hhi.add_trace(go.Scatter(
            x=cr_df["date"], y=cr_df["hhi_long"],
            mode="lines+markers", name="HHI 多头",
            line=dict(color="#00E676", width=1.5),
        ))
        fig_hhi.add_trace(go.Scatter(
            x=cr_df["date"], y=cr_df["hhi_short"],
            mode="lines+markers", name="HHI 空头",
            line=dict(color="#FF1744", width=1.5),
        ))
        fig_hhi.update_layout(
            template="plotly_dark", height=220, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
            font=dict(color="#CCCCCC", size=10),
            xaxis=dict(gridcolor="#333333"),
            yaxis=dict(gridcolor="#333333", title="HHI"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.caption("**Herfindahl 指数** — 比 CR 更全面的集中度度量")
        st.plotly_chart(fig_hhi, use_container_width=True, key="hhi_chart")

    with sub_col2:
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Bar(
            x=cr_df["date"], y=cr_df["total_long"],
            name="总多单", marker_color="#00E676", opacity=0.7,
        ))
        fig_pos.add_trace(go.Bar(
            x=cr_df["date"], y=cr_df["total_short"],
            name="总空单", marker_color="#FF1744", opacity=0.7,
        ))
        fig_pos.update_layout(
            template="plotly_dark", height=220, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
            font=dict(color="#CCCCCC", size=10),
            xaxis=dict(gridcolor="#333333"),
            yaxis=dict(gridcolor="#333333", title="持仓量"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            barmode="group",
        )
        st.caption("**总持仓量变化** — 持仓激增/骤降是结构变化的领先指标")
        st.plotly_chart(fig_pos, use_container_width=True, key="pos_chart")


def _render_structure_events(events: list, cr_df: pd.DataFrame):
    """Render structure change events as cards."""
    if not events:
        st.info("当前数据范围内未检测到显著的结构变化事件。\n\n"
                "结构事件的触发条件：\n"
                "- CR 趋势从收紧转为松动（或反之）\n"
                "- 多空 CR 交叉\n"
                "- HHI 与 CR 背离\n"
                "- 总持仓量骤变（>20%）")
        return

    severity_cn = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低"}
    type_cn = {
        "long_loosening": "多头松动",
        "short_loosening": "空头松动",
        "long_tightening": "多头收紧",
        "short_tightening": "空头收紧",
        "cr_cross": "CR 交叉",
        "hhi_divergence": "HHI 背离",
        "position_surge": "持仓异动",
    }

    for event in events[-20:][::-1]:
        with st.container(border=True):
            ecols = st.columns([3, 1, 1])
            with ecols[0]:
                event_type = type_cn.get(event["type"], event["type"])
                st.markdown(f"**{event_type}** — {event['date']}")
                st.caption(event["detail"])
            with ecols[1]:
                st.markdown(severity_cn.get(event["severity"], event["severity"]))
            with ecols[2]:
                side_label = {"long": "多头侧", "short": "空头侧", "both": "双向"}.get(
                    event.get("side", ""), "")
                st.caption(side_label)


def _render_broker_changes(stability, hist, symbol: str):
    """Render broker rank shake-up events."""
    if stability is None or stability.empty:
        st.info("席位排名数据不足（需 ≥2 个交易日）。")
        return

    shakeups = stability[stability["long_shakeup"] | stability["short_shakeup"]]

    if shakeups.empty:
        st.success("席位 Top-3 排名稳定，未检测到明显变动。")
    else:
        st.warning(f"检测到 **{len(shakeups)}** 次席位排名震荡事件")

    # Show all stability rows
    st.caption("**席位排名 Jaccard 稳定性**（1.0 = 完全不变, 0.0 = 完全替换）")
    fig_stab = go.Figure()
    fig_stab.add_trace(go.Scatter(
        x=stability["date"], y=stability["long_stability"],
        mode="lines+markers", name="多头前列稳定性",
        line=dict(color="#00E676", width=1.5),
    ))
    fig_stab.add_trace(go.Scatter(
        x=stability["date"], y=stability["short_stability"],
        mode="lines+markers", name="空头前列稳定性",
        line=dict(color="#FF1744", width=1.5),
    ))
    fig_stab.add_hline(y=0.5, line_dash="dash", line_color="#FFA726",
                       annotation_text="震荡阈值")
    fig_stab.update_layout(
        template="plotly_dark", height=220, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=10),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333", tickformat=".0%", range=[0, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_stab, use_container_width=True, key="stability_chart")

    # Show shakeup details
    for _, row in shakeups.iterrows():
        with st.container(border=True):
            date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])
            st.caption(f"**{date_str}**")
            scols = st.columns(2)
            with scols[0]:
                if row["long_shakeup"]:
                    st.markdown("🔓 **多头前列变动**")
                    new_b = row.get("new_long_brokers", [])
                    drop_b = row.get("dropped_long_brokers", [])
                    if new_b:
                        st.caption(f"▲ 新进: {', '.join(new_b)}")
                    if drop_b:
                        st.caption(f"▼ 退出: {', '.join(drop_b)}")
            with scols[1]:
                if row["short_shakeup"]:
                    st.markdown("🔓 **空头前列变动**")
                    new_b = row.get("new_short_brokers", [])
                    drop_b = row.get("dropped_short_brokers", [])
                    if new_b:
                        st.caption(f"▲ 新进: {', '.join(new_b)}")
                    if drop_b:
                        st.caption(f"▼ 退出: {', '.join(drop_b)}")


def _render_defection_events(defections: list, symbol: str):
    """Render broker defection events — summary + grouped by type."""
    if defections is None or len(defections) == 0:
        st.info(
            f"当前**{symbol.upper()}** 数据范围内未检测到席位叛变事件。\n\n"
            f"叛变检测类型：\n"
            f"- **方向翻转**：净多 ↔ 净空\n"
            f"- **大幅减仓**：净持仓缩减 >30%\n"
            f"- **动量逆转**：连续加仓方向突然反转\n\n"
            f"需要 ≥3 天的连续席位数据。"
        )
        return

    type_cfg = {
        "flip_long_to_short": ("多翻空", "🔴", "#FF1744", "净多→净空，最强烈的看空信号"),
        "flip_short_to_long": ("空翻多", "🟢", "#00E676", "净空→净多，最强烈的看多信号"),
        "long_unwind":        ("多头减仓", "🟠", "#FFA726", "多头大幅撤离，集中度松动"),
        "short_unwind":       ("空头减仓", "🟠", "#FFA726", "空头大幅回补，集中度松动"),
        "momentum_reversal":  ("动量逆转", "🟡", "#FFEB3B", "连续加仓方向反转"),
    }

    # ── Group events by type ──
    grouped = {}
    for e in defections:
        t = e["type"]
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(e)

    # Sort groups: flips first, then unwinds, then momentum
    type_order = [
        "flip_long_to_short", "flip_short_to_long",
        "long_unwind", "short_unwind", "momentum_reversal",
    ]
    ordered_groups = [(t, grouped[t]) for t in type_order if t in grouped]

    # ── Summary row ──
    summary_cols = st.columns(len(ordered_groups) + 1)
    high_count = sum(1 for e in defections if e.get("severity") == "high")
    with summary_cols[0]:
        st.metric("总事件", len(defections),
                  delta=f"🔴高 {high_count} | 🟡中 {len(defections)-high_count}",
                  delta_color="off")

    for i, (etype, evts) in enumerate(ordered_groups):
        label, emoji, color, _ = type_cfg.get(etype, (etype, "", "#ccc", ""))
        unique_brokers = len(set(e["broker"] for e in evts))
        with summary_cols[i + 1]:
            st.metric(
                f"{emoji} {label}",
                f"{len(evts)}次",
                delta=f"{unique_brokers}个席位",
                delta_color="off",
            )

    # ── Date-grouped event feed (primary view) ──
    st.divider()
    st.caption("**叛变日志** — 按日期分组，最近事件在上")

    _render_defection_feed(defections, type_cfg)

    # ── Alternative: bubble chart (collapsed) ──
    with st.expander("📈 气泡时间轴（辅助视图）", expanded=False):
        _render_defection_timeline(defections, type_cfg)

    # ── Type-grouped detail (collapsed) ──
    with st.expander("📋 按类型分组明细", expanded=False):
        for etype, evts in ordered_groups:
            label, emoji, color, desc = type_cfg.get(etype, (etype, "", "#ccc", ""))
            evts_sorted = sorted(
                evts,
                key=lambda e: (e["date"], {"high": 0, "medium": 1, "low": 2}.get(e.get("severity"), 9)),
                reverse=True,
            )
            st.caption(f"{emoji} **{label}** — {len(evts)} 次 | {desc}")
            for event in evts_sorted[:10]:
                severity_badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    event.get("severity", ""), "")
                st.markdown(
                    f"<span style='color:{color}'>**{event['broker']}**</span> "
                    f"{severity_badge} — {event['date']}",
                    unsafe_allow_html=True,
                )
                st.caption(event["detail"])


def _render_defection_feed(defections: list, type_cfg: dict):
    """Render defection events as a date-grouped feed — like a git log.

    Most recent dates first. Each date is a header. Each event is a
    compact card with severity, type badge, broker, and metrics.
    """
    # Group events by date (descending)
    from collections import defaultdict
    by_date = defaultdict(list)
    for e in defections:
        by_date[e["date"]].append(e)

    sorted_dates = sorted(by_date.keys(), reverse=True)

    severity_cfg = {
        "high": ("🔴 高", "#FF1744", 3),
        "medium": ("🟡 中", "#FFA726", 2),
        "low": ("🟢 低", "#4CAF50", 1),
    }

    for date_str in sorted_dates:
        day_events = by_date[date_str]
        # Sort within day: high severity first, then by type priority
        type_priority = {
            "flip_long_to_short": 0, "flip_short_to_long": 0,
            "long_unwind": 1, "short_unwind": 1, "momentum_reversal": 2,
        }
        day_events.sort(key=lambda e: (
            -severity_cfg.get(e.get("severity", "low"), ("", "", 0))[2],
            type_priority.get(e.get("type", ""), 9),
        ))

        high_in_day = sum(1 for e in day_events if e.get("severity") == "high")

        with st.container(border=True):
            # Date header
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown(f"### 📅 {date_str}")
            with header_cols[1]:
                st.caption(f"{len(day_events)} 起事件" +
                          (f" | {high_in_day} 高危" if high_in_day else ""))

            # Event cards within this date
            for event in day_events:
                etype = event.get("type", "")
                label, emoji, color, _ = type_cfg.get(etype, (etype, "", "#ccc", ""))
                sev_label, sev_color, _ = severity_cfg.get(
                    event.get("severity", "medium"), ("", "#888", 0))
                metrics = event.get("metrics", {})

                # Build compact metric string
                metric_parts = []
                dnp = metrics.get("delta_np", 0)
                if dnp != 0:
                    metric_parts.append(f"ΔNP: **{dnp:+}**")
                shrink = metrics.get("shrink_pct", 0)
                if shrink > 0:
                    metric_parts.append(f"减仓: **{shrink:.0%}**")
                net_before = metrics.get("net_pos_before")
                net_after = metrics.get("net_pos_after")
                if net_before is not None and net_after is not None:
                    metric_parts.append(f"{net_before:+} → {net_after:+}")

                cols = st.columns([1, 3, 2])
                with cols[0]:
                    # Severity + type badge
                    st.markdown(
                        f"<span style='font-size:1.3em'>{emoji}</span> "
                        f"<span style='color:{color};font-weight:bold'>{label}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(sev_label)

                with cols[1]:
                    st.markdown(f"**{event['broker']}**")
                    # Clean detail text (strip markdown bold)
                    detail = event.get("detail", "").replace("**", "")
                    # Truncate to first sentence
                    if "。" in detail:
                        detail = detail.split("。")[0] + "。"
                    st.caption(detail[:120])

                with cols[2]:
                    for mp in metric_parts:
                        st.caption(mp)

            # Show broker summary for this date
            day_brokers = sorted(set(e["broker"] for e in day_events))
            st.caption(f"涉及席位: {' · '.join(day_brokers)}")


def _render_defection_timeline(defections: list, type_cfg: dict):
    """Render a bubble-chart timeline of broker defection events.

    X = date, Y = broker, bubble size = severity × magnitude,
    color = event type.
    """
    import plotly.graph_objects as go
    import numpy as np

    # Build data for each event
    records = []
    for e in defections:
        label, emoji, color, _ = type_cfg.get(e["type"], (e["type"], "", "#ccc", ""))
        metrics = e.get("metrics", {})

        # Bubble size: severity is the PRIMARY driver, magnitude modulates it
        sev_mult = {"high": 3.0, "medium": 1.8, "low": 1.0}
        sev_base = sev_mult.get(e.get("severity", "medium"), 1.5)

        abs_dnp = abs(metrics.get("delta_np", 0))
        shrink_pct = metrics.get("shrink_pct", 0)

        # Combine severity + magnitude: base 12px × severity multiplier × magnitude factor
        if abs_dnp > 0:
            mag_factor = np.log1p(abs_dnp) / np.log1p(5000)  # normalized to ~5000 baseline
            mag_factor = max(0.5, min(mag_factor, 2.5))
        elif shrink_pct > 0:
            mag_factor = max(0.5, min(shrink_pct * 3, 2.5))
        else:
            mag_factor = 1.0

        bubble_size = 10 * sev_base * mag_factor

        records.append({
            "date": e["date"],
            "broker": e["broker"],
            "type_label": label,
            "type_emoji": emoji,
            "color": color,
            "size": max(bubble_size, 6),
            "severity": e.get("severity", "medium"),
            "detail": e.get("detail", ""),
            "delta_np": metrics.get("delta_np", 0),
            "shrink_pct": shrink_pct,
            "sev_mult": sev_base,
        })

    if not records:
        return

    # Sort records by date
    records.sort(key=lambda r: r["date"])

    # Unique brokers (sorted by first appearance) and dates
    seen_brokers = []
    for r in records:
        if r["broker"] not in seen_brokers:
            seen_brokers.append(r["broker"])

    all_dates = sorted(set(r["date"] for r in records))

    # Build one trace per event type for the legend
    type_traces = {}
    for r in records:
        t = r["type_label"]
        if t not in type_traces:
            type_traces[t] = {"x": [], "y": [], "size": [], "color": r["color"],
                              "emoji": r["type_emoji"], "sev_mult": r["sev_mult"],
                              "hover_date": [], "hover_broker": [],
                              "hover_detail": [], "hover_severity": [],
                              "hover_dnp": [], "hover_shrink": []}
        d = type_traces[t]
        d["x"].append(r["date"])
        d["y"].append(r["broker"])
        d["size"].append(r["size"])
        d["hover_date"].append(r["date"])
        d["hover_broker"].append(r["broker"])
        d["hover_detail"].append(r["detail"].replace("**", ""))
        d["hover_severity"].append(r["severity"])
        d["hover_dnp"].append(r["delta_np"])
        d["hover_shrink"].append(r["shrink_pct"])

    fig = go.Figure()

    for label, d in type_traces.items():
        hovertemplate = (
            f"<b>%{{customdata[0]}}</b> — %{{customdata[1]}}<br>"
            f"{d['emoji']} {label} | 严重度: %{{customdata[2]}}<br>"
            f"日期: %{{x}}<br>"
            f"ΔNP: %{{customdata[3]:+}}<br>"
            f"减仓幅度: %{{customdata[4]:.0%}}<br>"
            f"<i>%{{customdata[5]}}</i><extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"],
            mode="markers",
            name=f"{d['emoji']} {label}",
            marker=dict(
                size=d["size"],
                color=d["color"],
                opacity=0.75,
                line=dict(width=1, color="white"),
                sizemode="area",
                sizeref=0.3,
            ),
            customdata=list(zip(
                d["hover_broker"], d["hover_date"],
                d["hover_severity"], d["hover_dnp"],
                d["hover_shrink"], d["hover_detail"],
            )),
            hovertemplate=hovertemplate,
        ))

    fig.update_layout(
        template="plotly_dark",
        height=max(280, len(seen_brokers) * 42 + 80),
        margin=dict(l=10, r=20, t=30, b=10),
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=11),
        xaxis=dict(
            gridcolor="#333333", title=None,
            type="date", tickformat="%m/%d",
            range=[all_dates[0], all_dates[-1]] if len(all_dates) >= 2 else None,
        ),
        yaxis=dict(
            gridcolor="#333333", title=None,
            categoryorder="array",
            categoryarray=seen_brokers,
            autorange="reversed",  # top-to-bottom = first-to-last
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        hovermode="closest",
    )

    # Add date range selector
    if len(all_dates) >= 4:
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="近3天", step="day", stepmode="backward"),
                    dict(count=7, label="近7天", step="day", stepmode="backward"),
                    dict(step="all", label="全部"),
                ]),
                bgcolor="#333333", activecolor="#555555",
                font=dict(color="#CCCCCC"),
            ),
        )

    st.plotly_chart(fig, use_container_width=True, key="defection_timeline")


def _render_smart_money(corr_df, smart_money: dict, symbol: str):
    """Render broker-price correlation leaderboard."""
    if corr_df is None or corr_df.empty or smart_money is None:
        st.info(
            f"无法分析**{symbol.upper()}** 的席位-价格相关性。\n\n"
            f"可能原因：\n"
            f"- 尚未获取日线价格数据\n"
            f"- 席位数据不足（需 ≥10 天重叠）\n\n"
            f"**获取价格数据：**\n"
            f"```python\n"
            f"from src.fetcher_akshare import fetch_and_store_prices\n"
            f"fetch_and_store_prices('{symbol}')\n"
            f"```"
        )
        return

    leading_long = smart_money.get("leading_long", [])
    contrarian = smart_money.get("contrarian", [])
    noisy = smart_money.get("noisy", [])

    # ── Summary metrics ──
    sum_cols = st.columns(4)
    with sum_cols[0]:
        st.metric("领先多头", len(leading_long),
                  delta="席位加多→价格上涨" if leading_long else None)
    with sum_cols[1]:
        st.metric("反向指标", len(contrarian),
                  delta="席位加多→价格下跌" if contrarian else None)
    with sum_cols[2]:
        st.metric("噪音席位", len(noisy))
    with sum_cols[3]:
        total = len(leading_long) + len(contrarian) + len(noisy)
        signal_pct = (len(leading_long) + len(contrarian)) / max(total, 1)
        st.metric("信号比例", f"{signal_pct:.0%}")

    # ── Leaderboard tabs ──
    lead_tab, contrarian_tab, full_tab = st.tabs([
        "🟢 领先多头席位", "🔴 反向/空头席位", "📋 全量排名",
    ])

    with lead_tab:
        if leading_long:
            _render_corr_leaderboard(leading_long, "long")
        else:
            st.info("未检测到领先多头席位（ΔNP 与未来涨幅正相关）。")

    with contrarian_tab:
        if contrarian:
            _render_corr_leaderboard(contrarian, "short")
        else:
            st.info("未检测到反向指标席位（ΔNP 与未来涨幅负相关）。")

    with full_tab:
        st.caption("**全量相关性排名**（按 |Pearson r| 降序）")
        st.dataframe(
            corr_df[[
                "broker", "lag", "pearson_r", "spearman_r",
                "hit_rate", "avg_impact", "total_obs", "is_smart_money"
            ]].rename(columns={
                "broker": "席位", "lag": "领先天数",
                "pearson_r": "Pearson r", "spearman_r": "Spearman ρ",
                "hit_rate": "命中率", "avg_impact": "平均影响",
                "total_obs": "样本数", "is_smart_money": "聪明钱",
            }).style.format({
                "Pearson r": "{:.3f}", "Spearman ρ": "{:.3f}",
                "命中率": "{:.1%}", "平均影响": "{:.4%}",
            }),
            use_container_width=True,
            hide_index=True,
        )


def _render_corr_leaderboard(brokers: list[dict], direction: str):
    """Render a single correlation leaderboard."""
    import plotly.graph_objects as go

    if not brokers:
        return

    # Bar chart of Pearson r
    names = [b["broker"] for b in brokers[:10]]
    values = [b["pearson_r"] for b in brokers[:10]]
    hit_rates = [b["hit_rate"] for b in brokers[:10]]

    bar_color = "#00E676" if direction == "long" else "#FF1744"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values,
        orientation="h",
        marker=dict(color=bar_color, opacity=0.8),
        text=[f"r={v:.3f} | 命中={h:.0%}" for v, h in zip(values, hit_rates)],
        textposition="outside",
        name="Pearson r",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(200, len(names) * 36),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font=dict(color="#CCCCCC", size=11),
        xaxis=dict(gridcolor="#333333", title="Pearson r (ΔNP vs 未来收益)"),
        yaxis=dict(gridcolor="#333333", autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"corr_bar_{direction}")

    # Detail cards
    for broker_info in brokers[:10]:
        with st.container(border=True):
            bcols = st.columns([2, 1, 1, 1])
            with bcols[0]:
                st.markdown(f"**{broker_info['broker']}**")
            with bcols[1]:
                st.metric("Pearson r", f"{broker_info['pearson_r']:.3f}")
            with bcols[2]:
                st.metric("命中率", f"{broker_info['hit_rate']:.1%}")
            with bcols[3]:
                st.metric("领先天数", f"{broker_info['lag']}天")
            st.caption(
                f"Spearman ρ = {broker_info['spearman_r']:.3f}  |  "
                f"平均影响 = {broker_info['avg_impact']:.4%}"
            )


# ═══════════════════════════════════════════════════════════════════
# 品种偏见总览
# ═══════════════════════════════════════════════════════════════════

BIAS_TAG = {
    "bullish":  ("▲ 看多", "#00E676"),
    "bearish":  ("▼ 看空", "#FF1744"),
    "neutral":  ("◆ 中性", "#FFA726"),
    "skip":     ("◇ 跳过", "#888888"),
}

BIAS_SORT = {"bullish": 0, "bearish": 1, "neutral": 2, "skip": 3}


def _bias_rows():
    """收拢所有品种的偏见数据为列表。"""
    rows = []
    for sym, info in MONITORED_SYMBOLS.items():
        b = st.session_state.bias_data.get(sym, {})
        if not b:
            continue
        rows.append({
            "sym": sym,
            "name": info["name"],
            "bias": b.get("bias", "skip"),
            "conviction": b.get("conviction", 0),
            "cr_long": b.get("cr_long", 0),
            "cr_short": b.get("cr_short", 0),
            "dnp": b.get("aggregate_delta_np", 0),
            "top_long_cr": b.get("top_long_cr_brokers", []),
            "top_short_cr": b.get("top_short_cr_brokers", []),
            "top_bullish": b.get("top_bullish", []),
            "top_bearish": b.get("top_bearish", []),
        })
    rows.sort(key=lambda r: (BIAS_SORT.get(r["bias"], 9), -r["conviction"]))
    return rows


def render_bias_overview():
    """所有品种席位偏见一览，按方向分组、确信度排序，展示席位明细。"""

    rows = _bias_rows()
    if not rows:
        st.info("暂无偏见数据。先运行 `python src/fetcher_akshare.py`。")
        return

    for r in rows:
        label, color = BIAS_TAG.get(r["bias"], ("?", "#888"))

        with st.container(border=True):
            # ── 第一行：品种 + 方向标签 + 确信度 ──
            top = st.columns([3, 2, 1])
            with top[0]:
                st.markdown(
                    f"**{r['sym']}** {r['name']}  "
                    f"<span style='color:{color};font-weight:bold;font-size:1.1em'>{label}</span>",
                    unsafe_allow_html=True,
                )
            with top[1]:
                st.caption(
                    f"多头集中度 {r['cr_long']:.0%}  |  空头集中度 {r['cr_short']:.0%}"
                )
            with top[2]:
                st.caption(f"确信度 **{r['conviction']:.0%}**")

            # ── 第二行：席位明细 ──
            detail_parts = []

            # 多头集中度排名前3
            if r["top_long_cr"]:
                detail_parts.append(f"**持多前三**: {', '.join(r['top_long_cr'][:3])}")
            # 空头集中度排名前3
            if r["top_short_cr"]:
                detail_parts.append(f"**持空前三**: {', '.join(r['top_short_cr'][:3])}")

            if detail_parts:
                st.caption("  |  ".join(detail_parts))

            # ── 第三行：动量方向 ──
            momentum_parts = []
            if r["top_bullish"]:
                momentum_parts.append(f"▲ 加多: {', '.join(r['top_bullish'][:3])}")
            if r["top_bearish"]:
                momentum_parts.append(f"▼ 加空: {', '.join(r['top_bearish'][:3])}")
            if r["dnp"] != 0:
                momentum_parts.append(f"ΔNP {r['dnp']:+}")

            if momentum_parts:
                st.caption("  |  ".join(momentum_parts))


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    render_sidebar()

    # ── 第一行：K线图 + 信号推送 ──
    left_col, right_col = st.columns([3, 2])

    with left_col:
        symbol = st.session_state.selected_symbol
        tf = st.session_state.timeframe
        st.subheader(f"{symbol.upper()} — {SYMBOL_NAMES.get(symbol, '')} ({tf})")
        render_chart(symbol, tf)

    with right_col:
        render_alert_feed()

    # ── 第二行：结构演变（CR 时间序列 + 松动检测） ──
    with st.expander("📈 持仓结构演变（" + st.session_state.selected_symbol.upper() + "）", expanded=False):
        render_structure_evolution()

    # ── 第三行：全品种结构热力图 ──
    with st.expander("🌡 全品种结构热力图", expanded=False):
        render_structure_heatmap()

    # ── 第四行：席位偏见总览（可折叠） ──
    with st.expander("📊 席位偏见总览（全部品种）", expanded=True):
        render_bias_overview()

    time.sleep(UI_REFRESH_INTERVAL_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
