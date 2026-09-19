import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CoilScan Pro", page_icon="📉", layout="wide")

# ==========================================
# 1. CONFIG & GLOBALS
# ==========================================
EXCHANGE_SUFFIX = ".NS"
INDICATOR_LENGTH = 20
PCT_LOOKBACK = 100
BATCH_SIZE = 80

# 1d_scan  : 2y daily bars  -> signals, ATR, historical hit-rate
# 15m_live : last 5 days 15m -> live price, today's VWAP for entry timing
DATA_CFG = {
    "1d_scan":  {"interval": "1d",  "period": "2y"},
    "1d":       {"interval": "1d",  "period": "5y"},
    "15m_live": {"interval": "15m", "period": "5d"},
}

# Daily flat-top base settings: (base candles, max base width %, max distance to ceiling %)
BASE_N, BASE_MAX_WIDTH, BASE_MAX_DIST = 5, 6.0, 1.5

FALLBACK = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "LT", "ITC", "AXISBANK", "BHARTIARTL"]
UNIVERSES = ["All NSE (EQ series)", "Nifty 500", "Custom list"]


def _get_csv(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return pd.read_csv(r)


@st.cache_data(ttl=86400)
def fetch_universe(kind: str):
    """Returns (symbols, label). Falls back gracefully if NSE blocks the download."""
    try:
        if kind == "All NSE (EQ series)":
            df = _get_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
            df.columns = [c.strip() for c in df.columns]
            # EQ only: BE/BZ are trade-to-trade / restricted, bad for 1-2 day trades
            df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
            syms = [str(s).strip() for s in df["SYMBOL"] if pd.notna(s)]
            if len(syms) > 500:
                return syms, f"NSE all EQ ({len(syms)})"
        df = _get_csv("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
        syms = [str(s).strip() for s in df["Symbol"] if pd.notna(s) and "DUMMY" not in str(s)]
        if syms:
            return syms, f"Nifty 500 ({len(syms)})"
    except Exception:
        pass
    return FALLBACK, "Fallback list (NSE download failed)"


# ==========================================
# 2. INDICATORS
# ==========================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def pine_rising(s, n):
    return len(s) >= n + 1 and bool((s.diff().iloc[-n:] > 0).all())


def pine_falling(s, n):
    return len(s) >= n + 1 and bool((s.diff().iloc[-n:] < 0).all())


def compute_envelope(df, length=20):
    close = df["Close"].astype(float)
    basis = ema(close, length)
    d = ema((close - basis).abs(), length)
    upper, lower = basis + d, basis - d
    smooth = ema(pd.concat([upper, close], axis=1).max(axis=1), length)
    smooth2 = ema(pd.concat([close, lower], axis=1).min(axis=1), length)
    return pd.DataFrame({"close": close, "range": smooth - smooth2, "smooth": smooth, "smooth2": smooth2}, index=df.index)


def tick(x):
    return round(round(x / 0.05) * 0.05, 2)


# ==========================================
# 3. CORE PATTERN LOGIC (squeeze + flat-top base)
# ==========================================
def base_pattern(df, min_touches):
    N, max_width, max_dist = BASE_N, BASE_MAX_WIDTH, BASE_MAX_DIST
    env = compute_envelope(df, INDICATOR_LENGTH)
    close = env["close"]
    last_close = float(close.iloc[-1])

    rng_pct = env["range"] / close * 100
    last_rp = float(rng_pct.iloc[-1])
    hist = rng_pct.iloc[-PCT_LOOKBACK - 1:-1].dropna()
    pct_rank = float((hist < last_rp).mean() * 100) if len(hist) >= 30 else np.nan

    k = max(3, INDICATOR_LENGTH // 4)
    contracting = bool(last_rp < float(rng_pct.iloc[-1 - k]) and (rng_pct.diff().iloc[-k:] < 0).mean() >= 0.6)
    wl = max(1, INDICATOR_LENGTH // 5)
    wedge = bool(pine_rising(env["smooth2"], wl) and pine_falling(env["smooth"], wl))

    # Base = PRIOR N bars (excludes current bar so a breakout candle can't stretch its own base)
    base = df.iloc[-N - 1:-1]
    ceiling, floor = float(base["High"].max()), float(base["Low"].min())
    width_pct = (ceiling - floor) / floor * 100
    tight = width_pct <= max_width
    dist_pct = (ceiling - last_close) / ceiling * 100
    near = 0 <= dist_pct <= max_dist
    touches = int((base["High"] >= ceiling * (1 - max_dist / 200)).sum())
    half = max(1, N // 2)
    higher_lows = bool(base["Low"].iloc[half:].min() >= base["Low"].iloc[:half].min() * 0.998)

    is_setup = bool(tight and near and touches >= min_touches and higher_lows)
    is_breakout = bool(tight and touches >= min_touches and 0 < -dist_pct <= max_dist * 2)
    pattern = "Breakout" if is_breakout else ("Setup" if is_setup else "")

    vol_dryup, rvol = np.nan, np.nan
    if df["Volume"].sum() > 0:
        prior = df["Volume"].iloc[-N - 21:-N - 1].mean()
        if prior and prior > 0:
            vol_dryup = float(base["Volume"].mean() / prior)
        avg20 = df["Volume"].iloc[-21:-1].mean()
        if avg20 and avg20 > 0:
            rvol = float(df["Volume"].iloc[-1] / avg20)

    return {
        "pct_rank": pct_rank, "contracting": contracting, "wedge": wedge, "pattern": pattern,
        "ceiling": ceiling, "floor": floor, "width_pct": width_pct, "dist_pct": dist_pct,
        "touches": touches, "vol_dryup": vol_dryup, "rvol": rvol, "base_env": env,
    }


# ==========================================
# 4. FULL SIGNAL ENGINE + TRADE PLAN
# ==========================================
def analyze_pro(df, target, min_touches, drop_last=False):
    d = df.dropna(subset=["Open", "High", "Low", "Close"])
    if drop_last:
        d = d.iloc[:-1]
    if len(d) < 150:
        return None

    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    last = float(c.iloc[-1])
    a = float(atr(d).iloc[-1])
    if not a or np.isnan(a):
        return None

    bp = base_pattern(d, min_touches)
    rvol = bp["rvol"]

    # ---- Trend / momentum ----
    e9, e21, e50, e200 = ema(c, 9), ema(c, 21), ema(c, 50), ema(c, 200)
    rs = float(rsi(c).iloc[-1])
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    adx_s, pdi, mdi = adx(d)
    adx_v, pdi_v, mdi_v = float(adx_s.iloc[-1]), float(pdi.iloc[-1]), float(mdi.iloc[-1])
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_up, obv_dn = obv.iloc[-1] > obv.iloc[-10], obv.iloc[-1] < obv.iloc[-10]

    # ---- Range / compression signals ----
    bbw = (4 * c.rolling(20).std() / c.rolling(20).mean() * 100)
    bbw_hist = bbw.iloc[-121:-1].dropna()
    bb_pct = float((bbw_hist < bbw.iloc[-1]).mean() * 100) if len(bbw_hist) > 30 else np.nan
    rngs = (h - l)
    nr7 = bool(rngs.iloc[-1] <= rngs.iloc[-7:].min())
    inside = bool(h.iloc[-1] < h.iloc[-2] and l.iloc[-1] > l.iloc[-2])

    # ---- Breakout context ----
    hi20, lo20 = float(h.iloc[-21:-1].max()), float(l.iloc[-21:-1].min())
    hi252 = float(h.iloc[-253:-1].max())
    lo252 = float(l.iloc[-253:-1].min())
    brk20, brd20 = last > hi20, last < lo20
    near52 = last >= 0.92 * hi252

    # ---- Candlestick patterns (last 2 bars) ----
    o1, h1, l1, c1 = float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), last
    o2, c2 = float(o.iloc[-2]), float(c.iloc[-2])
    body = abs(c1 - o1) + 1e-9
    upw, dnw = h1 - max(c1, o1), min(c1, o1) - l1
    bull_engulf = c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2
    bear_engulf = c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2
    hammer = dnw >= 2 * body and upw <= body
    shooting = upw >= 2 * body and dnw <= body
    strong_close_up = (h1 - l1) > 1.3 * a and (c1 - l1) / (h1 - l1 + 1e-9) >= 0.8 and c1 > o1
    strong_close_dn = (h1 - l1) > 1.3 * a and (h1 - c1) / (h1 - l1 + 1e-9) >= 0.8 and c1 < o1
    gap_up, gap_dn = o1 > float(h.iloc[-2]), o1 < float(l.iloc[-2])

    # ---- Historical reachability: how often did this stock actually move `target` points in 2 days? ----
    nxt_hi = pd.concat([h.shift(-1), h.shift(-2)], axis=1).max(axis=1)
    nxt_lo = pd.concat([l.shift(-1), l.shift(-2)], axis=1).min(axis=1)
    fu, fd = (nxt_hi - c).iloc[-252:-2], (c - nxt_lo).iloc[-252:-2]
    hit_up, hit_dn = float((fu >= target).mean() * 100), float((fd >= target).mean() * 100)

    # ---------------- BULL score ----------------
    bull, bl = 0.0, []

    def B(cond, pts, label=None):
        nonlocal bull
        if cond:
            bull += pts
            if label:
                bl.append(label)

    pr = bp["pct_rank"]
    if not np.isnan(pr):
        B(True, (100 - pr) / 100 * 15, "Vol squeeze" if pr <= 30 else None)
    B(bp["contracting"], 4, "Contracting")
    B(bp["wedge"], 4, "Coil wedge")
    B(not np.isnan(bb_pct) and bb_pct <= 20, 5, "BB squeeze")
    B(nr7, 4, "NR7")
    B(inside, 2, "Inside day")
    B(bp["pattern"] == "Setup", 12, "Flat-top setup")
    B(bp["pattern"] == "Breakout", 15, "Flat-top breakout")
    B(brk20, 8, "20D high breakout")
    B(near52, 4, "Near 52W high")
    B(last > e50.iloc[-1], 2)
    B(e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1], 4, "EMA 9>21>50")
    B(last > e200.iloc[-1], 2, "Above EMA200")
    B(55 <= rs <= 75, 5, f"RSI {rs:.0f}")
    B(rs > 80, -3)
    B(hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2], 5, "MACD rising")
    B(adx_v > 20 and pdi_v > mdi_v, 4, f"ADX {adx_v:.0f} +DI")
    B(not np.isnan(rvol) and rvol >= 1.5, 8, f"RVOL {rvol:.1f}x")
    B(not np.isnan(rvol) and 1.0 <= rvol < 1.5, 3)
    B(obv_up, 3, "OBV up")
    B(bull_engulf, 4, "Bull engulfing")
    B(hammer, 3, "Hammer")
    B(strong_close_up, 3, "Momentum bar")
    B(gap_up, 2, "Gap up")
    B(True, min(hit_up, 15) / 15 * 15, f"{hit_up:.0f}% hit-rate" if hit_up >= 5 else None)
    bull = float(min(100, max(0, bull)))

    # ---------------- BEAR score ----------------
    bear, brl = 0.0, []

    def S(cond, pts, label=None):
        nonlocal bear
        if cond:
            bear += pts
            if label:
                brl.append(label)

    if not np.isnan(pr):
        S(True, (100 - pr) / 100 * 10, "Vol squeeze" if pr <= 30 else None)
    S(last < bp["floor"], 10, "Base breakdown")
    S(brd20, 8, "20D low breakdown")
    S(last < e50.iloc[-1], 3)
    S(e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1], 4, "EMA 9<21<50")
    S(last < e200.iloc[-1], 2, "Below EMA200")
    S(25 <= rs <= 45, 5, f"RSI {rs:.0f}")
    S(hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2], 5, "MACD falling")
    S(adx_v > 20 and mdi_v > pdi_v, 4, f"ADX {adx_v:.0f} -DI")
    S(not np.isnan(rvol) and rvol >= 1.5 and last < c2, 8, f"RVOL {rvol:.1f}x down")
    S(obv_dn, 3, "OBV down")
    S(bear_engulf, 4, "Bear engulfing")
    S(shooting, 3, "Shooting star")
    S(strong_close_dn, 3, "Bearish momentum bar")
    S(gap_dn, 2, "Gap down")
    S(True, min(hit_dn, 15) / 15 * 15, f"{hit_dn:.0f}% hit-rate" if hit_dn >= 5 else None)
    bear = float(min(100, max(0, bear)))

    side = "BUY" if bull >= bear else "SELL"
    score = bull if side == "BUY" else bear
    sigs = bl if side == "BUY" else brl
    hit = hit_up if side == "BUY" else hit_dn

    # ---------------- TRADE PLAN ----------------
    low3, high3 = float(l.iloc[-3:].min()), float(h.iloc[-3:].max())
    ceiling, floor = bp["ceiling"], bp["floor"]
    if side == "BUY":
        entry = last if last >= ceiling else ceiling + 0.1 * a           # buy-stop above ceiling
        sl = max(min(floor, low3) - 0.1 * a, entry - 2 * a)              # below base/3-bar low, capped at 2 ATR
        risk = max(entry - sl, 0.6 * a)
        sl = entry - risk
        t1, t2 = entry + target, entry + target + 5
    else:
        entry = last if last <= floor else floor - 0.1 * a               # sell-stop below floor
        sl = min(max(ceiling, high3) + 0.1 * a, entry + 2 * a)
        risk = max(sl - entry, 0.6 * a)
        sl = entry + risk
        t1, t2 = max(entry - target, 0.05), max(entry - target - 5, 0.05)

    return {
        "last_close": round(last, 2), "side": side, "score": round(score, 1),
        "bull_score": round(bull, 1), "bear_score": round(bear, 1),
        "entry": tick(entry), "sl": tick(sl), "t1": tick(t1), "t2": tick(t2),
        "risk": round(risk, 2), "rr": round(target / risk, 2) if risk else np.nan,
        "hit_rate": round(hit, 1), "atr": round(a, 2), "atr_pct": round(a / last * 100, 2),
        "rvol": round(rvol, 2) if not np.isnan(rvol) else np.nan,
        "rsi": round(rs, 1), "pattern": bp["pattern"], "ceiling": round(ceiling, 2), "floor": round(floor, 2),
        "signals": ", ".join(sigs), "as_of": str(d.index[-1]),
    }


def decide(r):
    """Turns the plan + live price into an actionable instruction."""
    px, e, sl, a, vw = r["live_price"], r["entry"], r["sl"], r["atr"], r["vwap"]
    has_vw = pd.notna(vw)
    if r["side"] == "BUY":
        if px <= sl:
            return "❌ Invalidated (at/below SL)"
        if px >= e:
            if px > e + 0.75 * a:
                return f"⚠️ Extended - wait for dip to {e}"
            if has_vw and px < vw:
                return "🟡 Above trigger but under VWAP - wait"
            return "🟢 BUY NOW"
        if px >= e - 0.35 * a:
            return f"🟡 Buy-stop above {e}"
        return f"⏳ Watch - buy above {e}"
    else:
        if px >= sl:
            return "❌ Invalidated (at/above SL)"
        if px <= e:
            if px < e - 0.75 * a:
                return f"⚠️ Extended - wait for bounce to {e}"
            if has_vw and px > vw:
                return "🟡 Below trigger but over VWAP - wait"
            return "🔴 SELL NOW (intraday short / exit longs)"
        if px <= e + 0.35 * a:
            return f"🟡 Sell-stop below {e}"
        return f"⏳ Watch - sell below {e}"


# ==========================================
# 5. DATA FETCHING
# ==========================================
def to_yahoo_symbol(s):
    return s if "." in s else f"{s}{EXCHANGE_SUFFIX}"


def _download(symbols, cfg_key):
    cfg = DATA_CFG[cfg_key]
    tickers = [to_yahoo_symbol(s) for s in symbols]
    raw = yf.download(tickers, interval=cfg["interval"], period=cfg["period"], auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    out = {}
    if raw is None or raw.empty:
        return out
    for sym, tk in zip(symbols, tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if tk not in raw.columns.get_level_values(0):
                    continue
                sub = raw[tk]
            else:
                sub = raw
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
            if not sub.empty:
                out[sym] = sub
        except Exception:
            continue
    return out


@st.cache_data(ttl=600, show_spinner=False)
def fetch_batch(symbols: tuple, cfg_key: str):
    return _download(symbols, cfg_key)


@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_batch(symbols: tuple):
    return _download(symbols, "15m_live")


def live_snapshot(symbols):
    """Latest price + today's VWAP from 15m bars (short cache so refresh really refreshes)."""
    out = {}
    symbols = list(symbols)
    for i in range(0, len(symbols), 40):
        data = fetch_live_batch(tuple(symbols[i:i + 40]))
        for s, d in data.items():
            d = d.dropna(subset=["Close"])
            if d.empty:
                continue
            day = d.index[-1].date()
            t = d[[x.date() == day for x in d.index]]
            vol = float(t["Volume"].sum())
            tp = (t["High"] + t["Low"] + t["Close"]) / 3
            vwap = float((tp * t["Volume"]).sum() / vol) if vol > 0 else float(t["Close"].mean())
            out[s] = {"live": float(t["Close"].iloc[-1]), "vwap": vwap,
                      "day_high": float(t["High"].max()), "day_low": float(t["Low"].min()),
                      "time": str(t.index[-1])}
    return out


def run_scan(symbols, target, min_touches, drop_last, pmin, pmax, min_turn_cr):
    rows, stats = [], {"no_data": 0, "price": 0, "liquidity": 0, "circuit": 0, "short_history": 0, "errors": 0}
    chunks = [tuple(symbols[i:i + BATCH_SIZE]) for i in range(0, len(symbols), BATCH_SIZE)]
    bar = st.progress(0)
    for ci, chunk in enumerate(chunks):
        try:
            data = fetch_batch(chunk, "1d_scan")
        except Exception:
            data = {}
        stats["no_data"] += len(chunk) - len(data)
        for sym, df in data.items():
            try:
                df = df.dropna(subset=["Close"])
                if len(df) < 150:
                    stats["short_history"] += 1
                    continue
                last = float(df["Close"].iloc[-1])
                if not (pmin <= last <= pmax):      # price filter first: cheap, removes most of the market
                    stats["price"] += 1
                    continue
                turnover = float((df["Close"] * df["Volume"]).iloc[-20:].mean())
                if turnover < min_turn_cr * 1e7:
                    stats["liquidity"] += 1
                    continue
                if df["High"].iloc[-1] == df["Low"].iloc[-1]:   # locked at circuit: can't trade
                    stats["circuit"] += 1
                    continue
                res = analyze_pro(df, target, min_touches, drop_last)
                if res and res["score"] >= 30:
                    res.update({"symbol": sym, "turnover_cr": round(turnover / 1e7, 1)})
                    rows.append(res)
            except Exception:
                stats["errors"] += 1
        bar.progress((ci + 1) / len(chunks))
    bar.empty()
    st.session_state.scan_stats = stats
    return pd.DataFrame(rows)


def refresh_live(scan, n):
    if scan.empty:
        return {}
    top = scan.sort_values("score", ascending=False).head(n)["symbol"].tolist()
    return live_snapshot(top)


def apply_live(scan, live):
    d = scan.copy()
    d["live_price"] = [live.get(s, {}).get("live", lc) for s, lc in zip(d["symbol"], d["last_close"])]
    d["vwap"] = [live.get(s, {}).get("vwap", np.nan) for s in d["symbol"]]
    d["Live"] = ["✅" if s in live else "—" for s in d["symbol"]]
    d["Action"] = d.apply(decide, axis=1)
    return d


# ==========================================
# 6. STREAMLIT UI
# ==========================================
st.title("📉 CoilScan Pro — Under ₹400 Momentum Scanner")
st.caption("Ranks stocks by squeeze + breakout + momentum + volume signals AND how often each stock has "
           "really moved your target points within 2 days. It ranks probability; it cannot guarantee any move. "
           "Always place the stop loss.")

for k, v in {"scan_data": pd.DataFrame(), "live": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.header("⚙️ Scan Settings")
    universe = st.selectbox("Universe", UNIVERSES)
    custom = st.text_area("Custom symbols (comma separated)", "") if universe == "Custom list" else ""
    c1, c2 = st.columns(2)
    pmin = c1.number_input("Min price ₹", 5.0, 400.0, 15.0)
    pmax = c2.number_input("Max price ₹", 20.0, 5000.0, 400.0)
    target = st.slider("Target points (2 days)", 15, 60, 30)
    min_turn = st.number_input("Min avg daily turnover (₹ Cr)", 0.5, 100.0, 5.0)
    min_touches = st.slider("Min ceiling touches", 1, 4, 2)
    drop_last = st.checkbox("Ignore in-progress candle", False)
    max_syms = st.number_input("Max symbols (0 = all, use ~300 to test fast)", 0, 5000, 0)
    live_n = st.slider("Live-check top N stocks", 10, 100, 50)

    if st.button("🔄 Run Full Scan", type="primary", width="stretch"):
        if universe == "Custom list":
            syms = [s.strip().upper() for s in custom.replace("\n", ",").split(",") if s.strip()]
            label = f"Custom ({len(syms)})"
        else:
            syms, label = fetch_universe(universe)
        if max_syms:
            syms = syms[:int(max_syms)]
        if not syms:
            st.warning("No symbols to scan.")
        else:
            with st.spinner(f"Scanning {len(syms)} symbols ({label})..."):
                st.session_state.scan_data = run_scan(syms, target, min_touches, drop_last, pmin, pmax, min_turn)
                st.session_state.target = target
                st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.universe_label = label
            with st.spinner("Fetching live 15m prices & VWAP..."):
                st.session_state.live = refresh_live(st.session_state.scan_data, live_n)
                st.session_state.live_time = datetime.now().strftime("%H:%M:%S")

    if st.button("⚡ Refresh Live Prices Only", width="stretch"):
        with st.spinner("Refreshing..."):
            fetch_live_batch.clear()
            st.session_state.live = refresh_live(st.session_state.scan_data, live_n)
            st.session_state.live_time = datetime.now().strftime("%H:%M:%S")

    st.divider()
    st.header("🔍 Filters")
    sides = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"])
    min_score = st.slider("Min score", 0, 100, 50)
    min_hit = st.slider("Min 2-day hit-rate %", 0, 30, 3)
    min_rr = st.slider("Min reward:risk", 0.5, 5.0, 1.5, 0.1)
    only_actionable = st.checkbox("Only actionable (Buy/Sell now or stop-orders)", False)
    search_q = st.text_input("Search symbol")

scan = st.session_state.scan_data

if scan.empty:
    st.info("👈 Click **Run Full Scan**. A full-market scan takes a few minutes; later use **Refresh Live Prices** during the day.")
else:
    stt = st.session_state.get("scan_stats", {})
    st.caption(
        f"Scan: {st.session_state.get('last_run')} · {st.session_state.get('universe_label')} · "
        f"live prices: {st.session_state.get('live_time', '—')} ({len(st.session_state.live)} stocks live-checked) · "
        f"skipped → price: {stt.get('price', 0)}, liquidity: {stt.get('liquidity', 0)}, "
        f"circuit: {stt.get('circuit', 0)}, no data: {stt.get('no_data', 0)}"
    )

    view = apply_live(scan, st.session_state.live)
    view = view[view["side"].isin(sides) & (view["score"] >= min_score)
                & (view["hit_rate"] >= min_hit) & (view["rr"] >= min_rr)]
    if search_q:
        view = view[view["symbol"].str.contains(search_q, case=False)]
    if only_actionable:
        view = view[view["Action"].str.contains("NOW|Buy-stop|Sell-stop", regex=True)]
    view = view.sort_values(["score", "hit_rate"], ascending=False)

    if view.empty:
        st.warning("Nothing passes the filters right now. Lower min score / hit-rate, or scan again after the market moves. "
                   "A 30-point move on a sub-₹400 stock is 8-15%, so genuine candidates are few.")
    else:
        tgt = st.session_state.get("target", target)
        out = view[["symbol", "Action", "side", "live_price", "entry", "sl", "t1", "t2", "risk", "rr", "score",
                    "hit_rate", "atr", "rvol", "pattern", "signals", "Live"]].copy()
        out.columns = ["Symbol", "Action", "Side", "Price", "Entry", "Stop Loss", f"T1 (+{tgt})", f"T2 (+{tgt + 5})",
                       "Risk pts", "R:R", "Score", "2D Hit %", "ATR", "RVOL", "Pattern", "Signals", "Live"]
        st.dataframe(out, width="stretch", hide_index=True)
        st.caption("Entry = buy-stop / sell-stop trigger (or current price if already triggered). "
                   "Suggested handling: book half at halfway to T1, move SL to entry after that, exit the rest at T1/T2. "
                   "SELL in cash segment is intraday only (MIS); for overnight shorts you need F&O. "
                   "RVOL looks low early in the session because volume is still building.")

        # ---------------- CHART ----------------
        st.divider()
        st.subheader("📊 Trade Plan Chart")
        from lightweight_charts.widgets import StreamlitChart

        sym = st.selectbox("Symbol", view["symbol"].tolist())
        row = view[view["symbol"] == sym].iloc[0]

        with st.spinner("Loading chart..."):
            cdf = fetch_batch((sym,), "1d").get(sym, pd.DataFrame())
            if not cdf.empty:
                env = compute_envelope(cdf, INDICATOR_LENGTH)
                tv = cdf.reset_index()
                tv = tv.rename(columns={tv.columns[0]: "time", "Open": "open", "High": "high",
                                        "Low": "low", "Close": "close", "Volume": "volume"})
                t = pd.to_datetime(tv["time"])
                if t.dt.tz is not None:
                    t = t.dt.tz_localize(None)
                tv["time"] = t.dt.strftime("%Y-%m-%d")

                chart = StreamlitChart(width=900, height=550)
                chart.set(tv)

                def add_line(name, series, color, width=2):
                    ln = chart.create_line(name=name, color=color, width=width)
                    ln.set(pd.DataFrame({"time": tv["time"], name: series.values}).dropna())

                add_line("Upper Envelope", env["smooth"], "rgba(0,150,255,0.7)", 1)
                add_line("Lower Envelope", env["smooth2"], "rgba(0,150,255,0.7)", 1)

                span = 15
                def add_level(name, price, color):
                    ln = chart.create_line(name=name, color=color, width=2)
                    ln.set(pd.DataFrame({"time": tv["time"].iloc[-span:], name: [float(price)] * span}))

                add_level("Entry", row["entry"], "rgba(255,255,255,0.9)")
                add_level("Stop Loss", row["sl"], "rgba(255,0,0,0.9)")
                add_level("Target 1", row["t1"], "rgba(0,200,0,0.9)")
                add_level("Target 2", row["t2"], "rgba(0,255,120,0.9)")
                if row["pattern"]:
                    add_level("Ceiling", row["ceiling"], "rgba(255,165,0,0.9)")
                chart.load()
