
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BMonthEnd, BDay
from matplotlib.ticker import PercentFormatter

# =========================
# 0) Logging & constants
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
plt.rcParams["figure.dpi"] = 120

DEFAULT_INDIR  = Path("1_0_DATA_ENG")
DEFAULT_OUTDIR = Path("2_0_ALPHA_OUTPUTS")

# =========================
# 1) Utilities & loading
# =========================
def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def _check_cols(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required columns: {missing}")

def _read_best(path_base: Path) -> Optional[pd.DataFrame]:
    pq = path_base.with_suffix(".parquet")
    cs = path_base.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    return None

# --- Simple plotting percent axis helper ---
def _fmt_percent(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)

def _month_eom(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce").dt.to_period("M").dt.to_timestamp("M")

def _safe_parse_ticker_list(x) -> set:
    """Parse universe_monthly tickers robustly (JSON first, then literal_eval)."""
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return set([str(t).strip() for t in x])
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return set()
        # Try JSON
        try:
            obj = json.loads(s)
            return _safe_parse_ticker_list(obj)
        except Exception:
            pass
        # Try Python literal
        try:
            obj = ast.literal_eval(s)
            return _safe_parse_ticker_list(obj)
        except Exception:
            return set()
    return set()

def _load_exports(indir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    indir = Path(indir)

    # Primary daily panel (try several common names)
    df = None
    for stem in ["factors_daily", "panel_with_beta", "df_with_beta"]:
        df = _read_best(indir / stem)
        if df is not None:
            break
    if df is None:
        raise FileNotFoundError("Missing panel: factors_daily.*, panel_with_beta.*, or df_with_beta.* in DATA_ENG.")

    # EOM neutralized/z-scored (optional)
    eom_z = _read_best(indir / "factors_eom_z")

    # Market series (optional)
    mkt = _read_best(indir / "market_series")
    if mkt is None:
        mkt = _read_best(indir / "global_market_ew")

    # Factors list (optional)
    factors_list = None
    fl = indir / "factors_list.json"
    if fl.exists():
        try:
            factors_list = json.loads(fl.read_text(encoding="utf-8"))
        except Exception:
            factors_list = None

    # Last date (optional)
    last_date_txt = indir / "last_date.txt"
    if last_date_txt.exists():
        try:
            last_date = pd.to_datetime(last_date_txt.read_text().strip())
        except Exception:
            last_date = pd.to_datetime(df["date"]).max()
    else:
        last_date = pd.to_datetime(df["date"]).max()

    # Universe monthly (optional)
    uni = None
    uni_path = indir / "universe_monthly.csv"
    if uni_path.exists():
        try:
            uni = pd.read_csv(uni_path)
            uni["month_end"] = pd.to_datetime(uni["month_end"])
            uni["tickers"] = uni["tickers"].apply(_safe_parse_ticker_list)
        except Exception as e:
            logging.warning("Failed to load universe_monthly.csv: %s", e)

    return dict(df=df, eom_z=eom_z, mkt=mkt, factors_list=factors_list, last_date=last_date, universe=uni)

def clip_series(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    ql, qh = s.quantile([lo, hi])
    return s.clip(ql, qh)

def apply_embargo(df: pd.DataFrame, embargo_days: int = 1, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date_eff"] = df[date_col] + BDay(embargo_days)
    return df

def assert_monotone_by_ticker(df: pd.DataFrame, ticker_col="ticker", date_col="date") -> None:
    s = df[[ticker_col, date_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    bad = (
        s.sort_values([ticker_col, date_col])
         .assign(dt=lambda x: x.groupby(ticker_col)[date_col].diff().dt.days)
         .query("dt < 0")
    )
    if len(bad):
        tickers = bad[ticker_col].unique()
        raise ValueError(
            f"Found non-monotone dates within {len(tickers)} tickers "
            f"(examples: {', '.join(map(str, tickers[:5]))} ...)."
        )

# =========================
# 2) Forward returns & effective return
# =========================
def forward_return_from_daily_strict(ret: pd.Series, horizon_days: int) -> pd.Series:
    r = pd.to_numeric(ret, errors="coerce")
    out = (1.0 + r).rolling(horizon_days, min_periods=horizon_days).apply(
        lambda x: np.prod(x) - 1.0, raw=True
    )
    return out.shift(-horizon_days)

def add_effective_return(df: pd.DataFrame,
                         prefer_col="excess_ret_1d_eur",
                         fallback_col="ret_1d_eur",
                         out_col="ret_effective",
                         winsorize=True,
                         clip_lo=0.01, clip_hi=0.99) -> pd.DataFrame:
    if prefer_col not in df.columns and fallback_col not in df.columns:
        raise KeyError(f"Need at least one of {prefer_col!r} or {fallback_col!r} in df.")
    df = df.copy()
    pref = pd.to_numeric(df.get(prefer_col, np.nan), errors="coerce")
    fall = pd.to_numeric(df.get(fallback_col, np.nan), errors="coerce")
    if winsorize:
        pref = clip_series(pref, clip_lo, clip_hi)
        fall = clip_series(fall, clip_lo, clip_hi)
    df[out_col] = pref.where(pref.notna(), fall)
    return df

def add_forward_returns(df: pd.DataFrame, ret_col: str, horizons: List[int]) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        col = f"fwd_{ret_col}_{h}d"
        df[col] = (
            df.groupby("ticker", sort=False)[ret_col]
              .apply(lambda s: forward_return_from_daily_strict(s, h))
              .values
        )
    return df

# =========================
# 3) Cross-sectional IC engine
# =========================
def _xsec_corr(x: pd.Series, y: pd.Series, method: str = "pearson", min_n: int = 10) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < min_n:
        return np.nan
    a, b = s.iloc[:,0], s.iloc[:,1]
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    if method == "spearman":
        a = a.rank(method="average")
        b = b.rank(method="average")
        if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
            return np.nan
    return a.corr(b)

def compute_ic_timeseries(df: pd.DataFrame,
                          factor_cols: List[str],
                          fwd_ret_col: str,
                          *,
                          freq: str = "D",
                          min_names: int = 10,
                          method: str = "spearman",
                          date_col: str = "date",
                          use_embargo_date: bool = True) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    group_date = "date_eff" if use_embargo_date and "date_eff" in df.columns else date_col

    if freq.upper().startswith("M"):
        flags = df[date_col].dt.normalize() == (df[date_col].dt.to_period("M").dt.to_timestamp() + BMonthEnd(0))
        panel = df.loc[flags].copy()
    else:
        panel = df

    out_rows = []
    for d, g in panel.groupby(group_date, sort=True):
        n_total = g["ticker"].nunique()
        row = {"date": pd.to_datetime(d), "n": int(n_total)}
        if n_total < min_names:
            for fac in factor_cols:
                row[fac] = np.nan
            out_rows.append(row)
            continue
        for fac in factor_cols:
            ic_val = _xsec_corr(g[fac], g[fwd_ret_col], method=method, min_n=min_names)
            row[fac] = ic_val
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)

# =========================
# 4) Newey–West + FDR
# =========================
def newey_west_tstat(x: pd.Series, *, lags: int) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    n = len(x)
    if n < 5:
        return np.nan
    m = x.mean()
    xc = x - m
    gamma0 = np.dot(xc, xc) / n
    var = gamma0
    for L in range(1, min(lags, n-1) + 1):
        w = 1 - L / (lags + 1)
        gammaL = np.dot(xc[L:], xc[:-L]) / n
        var += 2 * w * gammaL
    se = math.sqrt(var / n)
    return m / se if se > 0 else np.nan

def _pnorm_two_sided_from_t(tval: float) -> float:
    if pd.isna(tval): return np.nan
    return math.erfc(abs(tval) / math.sqrt(2.0))

def fdr_bh(pvals: pd.Series) -> pd.Series:
    p = pvals.fillna(1).to_numpy(float)
    m = len(p)
    idx = np.argsort(p)
    ranked = p[idx]
    q = np.empty_like(p, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, m * ranked[i] / (i + 1))
        q[idx[i]] = prev
    return pd.Series(q, index=pvals.index)

def summarize_ic(ic_ts: pd.DataFrame,
                 factor_cols: List[str],
                 *,
                 freq: str = "D",
                 h_overlapping: int = 1) -> pd.DataFrame:
    ann = math.sqrt(252.0) if freq.upper().startswith("D") else math.sqrt(12.0)
    rows = []
    for fac in factor_cols:
        s = pd.to_numeric(ic_ts[fac], errors="coerce").dropna()
        if len(s) == 0:
            rows.append({"factor": fac, "mean": np.nan, "std": np.nan,
                         "IR_ann": np.nan, "hit_rate": np.nan, "n": 0, "t_NW": np.nan})
            continue
        mu  = s.mean()
        sd  = s.std(ddof=1)
        ir  = (mu / sd) * ann if sd and np.isfinite(sd) else np.nan
        lags = max(0, h_overlapping - 1)
        tnw = newey_west_tstat(s, lags=lags)
        rows.append({"factor": fac, "mean": mu, "std": sd,
                     "IR_ann": ir, "hit_rate": (s > 0).mean(), "n": int(s.count()), "t_NW": tnw})
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)

def summarize_ic_with_fdr(ic_ts: pd.DataFrame,
                          factor_cols: List[str],
                          *,
                          freq: str = "D",
                          h_overlapping: int = 1) -> pd.DataFrame:
    out = summarize_ic(ic_ts, factor_cols, freq=freq, h_overlapping=h_overlapping)
    out["p_NW"] = out["t_NW"].apply(_pnorm_two_sided_from_t)
    out["q_BH"] = fdr_bh(out["p_NW"])
    return out

# =========================
# 5) Neutralization & z
# =========================
def xsec_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(), s.std(ddof=0)
    return (s - m) / (sd if sd and np.isfinite(sd) else 1.0)

def xsec_robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    denom = 1.4826 * mad if mad and np.isfinite(mad) else s.std(ddof=0)
    denom = denom if denom and np.isfinite(denom) else 1.0
    return (s - med) / denom

def neutralize(df: pd.DataFrame,
               factor_col: str,
               by_cols=("country","supersector","size_bucket","liq_bucket")) -> pd.Series:
    out = pd.Series(index=df.index, dtype="float64")
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp[factor_col] = pd.to_numeric(tmp[factor_col], errors="coerce")
    use_cols = []
    for c in by_cols:
        if c in tmp.columns:
            tmp[c] = tmp[c].astype("category")
            use_cols.append(c)

    for d, g in tmp.groupby("date", sort=False):
        y = g[factor_col].to_numpy(dtype="float64")
        valid = np.isfinite(y)
        if valid.sum() < 3:
            out.loc[g.index] = np.nan
            continue
        g2 = g.loc[valid]
        y = y[valid]
        Xd = pd.get_dummies(g2[use_cols], drop_first=True, dtype=float) if use_cols else pd.DataFrame(index=g2.index)
        if Xd.shape[1] == 0:
            resid = y - np.nanmean(y)
        else:
            X = np.column_stack([np.ones(len(g2), dtype="float64"),
                                 Xd.to_numpy(dtype="float64", na_value=0.0)])
            lam = 1e-8
            beta = np.linalg.lstsq(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y, rcond=None)[0]
            resid = y - X @ beta
        out.loc[g2.index] = resid
    return out

def neutralize_and_z(df: pd.DataFrame,
                     factor: str,
                     by=("country","supersector","size_bucket","liq_bucket"),
                     robust: bool = False) -> Tuple[pd.Series, pd.Series]:
    resid = neutralize(df, factor, by_cols=by)
    zf = xsec_robust_z if robust else xsec_zscore
    z_neu = resid.groupby(df["date"]).transform(zf)
    z_raw = df.groupby("date")[factor].transform(zf)
    return z_raw, z_neu

# =========================
# 6) Weighting: IC-based (static), rolling IC, & ridge
# =========================
def ic_weights(ic_summary: pd.DataFrame, *, min_q: float = 0.2, shrink: float = 0.5) -> pd.Series:
    s = ic_summary.set_index("factor")["mean"].copy()
    s = s.replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    thr = s.abs().quantile(min_q)
    s = s.where(s.abs() >= thr, 0.0)
    if s.abs().sum() > 0:
        s = s / s.abs().sum()
    active = (s != 0)
    if active.any():
        ew = pd.Series(0.0, index=s.index)
        ew[active] = 1.0 / active.sum()
        s = shrink * ew + (1 - shrink) * s
    return s

def rolling_ic_weights(ic_m: pd.DataFrame,
                       factors: list[str],
                       lookback_months: int = 18,
                       shrink_to_ew: float = 0.6,
                       max_abs_w: float = 0.40,
                       min_months: int = 10,
                       freeze_when_thin: bool = True) -> pd.DataFrame:
    """
    Time-varying IC weights by month-end:
      - rolling mean IC over last L months
      - L1-normalize, clip to +/- max_abs_w
      - shrink toward equal weight
      - if sample < min_months: freeze previous row if available; else EW
    Returns DataFrame indexed by month-end with columns=factors.
    """
    if ic_m is None or ic_m.empty:
        return pd.DataFrame(columns=factors)

    use = [f for f in factors if f in ic_m.columns]
    if not use:
        return pd.DataFrame(columns=factors)

    ic = ic_m.copy()
    ic["date"] = pd.to_datetime(ic["date"]).dt.to_period("M").dt.to_timestamp("M")
    out_rows = []
    prev_w: Optional[pd.Series] = None

    for dt in sorted(ic["date"].unique()):
        hist = ic.loc[ic["date"] <= dt, use].tail(lookback_months)
        if hist.shape[0] < min_months:
            if freeze_when_thin and prev_w is not None:
                w = prev_w.copy()
            else:
                w = pd.Series(1.0 / len(use), index=use)
        else:
            m = hist.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if m.abs().sum() == 0:
                w = pd.Series(1.0 / len(use), index=use)
            else:
                w = m / m.abs().sum()
                w = w.clip(lower=-max_abs_w, upper=+max_abs_w)
                s = w.abs().sum()
                w = (w / s) if s > 0 else pd.Series(1.0 / len(use), index=use)
            ew = pd.Series(1.0 / len(use), index=use)
            w = shrink_to_ew * ew + (1 - shrink_to_ew) * w

        w.name = dt
        out_rows.append(w)
        prev_w = w.copy()

    W = pd.DataFrame(out_rows).sort_index()
    for f in factors:
        if f not in W.columns:
            W[f] = 0.0
    return W[factors]

def datewise_ridge_weights(df: pd.DataFrame, z_cols: List[str], fwd_col: str,
                           *, alpha: float = 2.0, min_n: int = 15) -> pd.Series:
    z_cols = [c for c in z_cols if c in df.columns]
    if len(z_cols) == 0:
        return pd.Series(dtype=float)
    W = []
    for d, g in df.groupby("date"):
        g = g[z_cols + [fwd_col]].dropna()
        if len(g) < min_n:
            continue
        X = g[z_cols].to_numpy(dtype=float)
        y = g[fwd_col].to_numpy(dtype=float)
        keep = X.std(axis=0, ddof=0) > 0
        if not keep.any():
            continue
        X = X[:, keep]
        XtX = X.T @ X
        try:
            beta = np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), X.T @ y)
        except np.linalg.LinAlgError:
            continue
        full_beta = np.full(len(z_cols), np.nan, dtype=float)
        j = 0
        for i, k in enumerate(keep):
            if k:
                full_beta[i] = beta[j]; j += 1
        W.append(full_beta)
    if not W:
        if len(z_cols) == 1:
            return pd.Series({z_cols[0]: 1.0})
    w = np.nanmean(np.vstack(W), axis=0) if W else np.ones(len(z_cols))
    w = np.where(np.isfinite(w), w, 0.0)
    ew = np.ones_like(w) / len(w)
    lam = 0.5
    w = lam * ew + (1 - lam) * w
    s = w.sum()
    if s and np.isfinite(s):
        w = w / s
    return pd.Series(w, index=z_cols)

# =========================
# 7) Portfolio construction (long-only & LS diagnostic)
# =========================
def _rebalance_flag(dates: pd.Series, how: str = "M") -> pd.Series:
    d = pd.to_datetime(dates)
    if how.upper().startswith("M"):
        return d.dt.normalize() == (d.dt.to_period("M").dt.to_timestamp() + pd.offsets.BMonthEnd(0))
    elif how.upper().startswith("W"):
        return d.dt.weekday == 4
    else:
        return pd.Series(True, index=dates.index)

def build_longonly_weights(df: pd.DataFrame, score_col: str, topq: float = 0.2, rebalance: str = "M") -> pd.DataFrame:
    need = {"date","ticker",score_col}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise KeyError(f"build_longonly_weights missing {missing}")

    x = df[["date","ticker",score_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values(["date","ticker"])

    x["is_reb"] = _rebalance_flag(x["date"], rebalance)

    ws = []
    for d, g in x[x["is_reb"]].groupby("date", sort=True):
        s = pd.to_numeric(g[score_col], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty:
            continue
        k = max(1, int(np.floor(len(s) * topq)))
        sel = s.nlargest(k).index
        w = pd.Series(0.0, index=g.index, name="w")
        w.loc[sel] = 1.0 / k
        out = g[["date","ticker"]].copy()
        out["w"] = w.values
        ws.append(out)

    wdf = pd.concat(ws, ignore_index=True) if ws else pd.DataFrame(columns=["date","ticker","w"])
    return wdf

def build_longshort_weights(df: pd.DataFrame, score_col: str, topq: float = 0.2, rebalance: str = "M") -> pd.DataFrame:
    need = {"date","ticker",score_col}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise KeyError(f"build_longshort_weights missing {missing}")

    x = df[["date","ticker",score_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values(["date","ticker"])
    x["is_reb"] = _rebalance_flag(x["date"], rebalance)

    ws = []
    for d, g in x[x["is_reb"]].groupby("date", sort=True):
        s = pd.to_numeric(g[score_col], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty:
            continue
        k = max(1, int(np.floor(len(s) * topq)))
        long_idx  = s.nlargest(k).index
        short_idx = s.nsmallest(k).index
        w = pd.Series(0.0, index=g.index, name="w")
        if len(long_idx):  w.loc[long_idx]  =  +0.5 / len(long_idx)
        if len(short_idx): w.loc[short_idx] =  -0.5 / len(short_idx)
        out = g[["date","ticker"]].copy()
        out["w"] = w.values
        ws.append(out)
    return pd.concat(ws, ignore_index=True) if ws else pd.DataFrame(columns=["date","ticker","w"])

def realize_portfolio_returns(df: pd.DataFrame, w_reb: pd.DataFrame,
                              ret_col: str = "ret_1d_eur", tc_bps: float = 0.0) -> pd.Series:
    """
    Realize daily portfolio returns by forward-filling the latest rebalance weights.
    - Costs applied once on the first trading day after each rebalance date.
    - Uses NEXT trading day's returns after the rebalance (no same-day lookahead).
    """
    need = {"date","ticker",ret_col}
    if not need.issubset(df.columns):
        raise KeyError(f"realize_portfolio_returns: df needs {need}")

    if w_reb is None or w_reb.empty:
        return pd.Series(dtype=float)

    px = df[["date","ticker",ret_col]].copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"]).sort_values(["date","ticker"])

    w = w_reb[["date","ticker","w"]].copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"]).sort_values(["date","ticker"])

    rebs = (w[["date"]].drop_duplicates().sort_values("date")
            .rename(columns={"date":"reb_id"}))
    px = pd.merge_asof(px.sort_values("date"),
                       rebs.sort_values("reb_id"),
                       left_on="date", right_on="reb_id",
                       direction="backward")
    px = px[px["reb_id"].notna()].copy()
    px = px[px["date"] > px["reb_id"]]

    ww = w.rename(columns={"date":"reb_id"})
    px = px.merge(ww, on=["reb_id","ticker"], how="left")
    px["w"] = px["w"].fillna(0.0)

    px["ret_contrib"] = px[ret_col] * px["w"]
    port = (px.groupby(["date","reb_id"], as_index=False)["ret_contrib"]
              .sum()
              .rename(columns={"ret_contrib":"ret_port"})
              .sort_values("date"))

    if tc_bps and tc_bps != 0:
        tc = tc_bps / 1e4  # cost per 100% turnover
        w_by_reb = {d: g.set_index("ticker")["w"] for d, g in w.groupby("reb_id" if "reb_id" in w.columns else "date", sort=True)}
        turns = []
        prev_vec = None
        for d in sorted({k for k in w_by_reb.keys()}):
            vec = w_by_reb[d]
            if prev_vec is None:
                trn = 0.0
            else:
                keys = prev_vec.index.union(vec.index)
                trn = float(np.abs(prev_vec.reindex(keys, fill_value=0.0) - vec.reindex(keys, fill_value=0.0)).sum())
            turns.append((d, trn))
            prev_vec = vec
        turn_df = pd.DataFrame(turns, columns=["reb_id","turnover_frac"])
        first_days = (port.groupby("reb_id", as_index=False)["date"].min()
                           .merge(turn_df, on="reb_id", how="left"))
        first_days["tc_ret"] = - first_days["turnover_frac"].fillna(0.0) * tc
        port = port.merge(first_days[["reb_id","date","tc_ret"]],
                          on=["reb_id","date"], how="left")
        port["ret_port"] = port["ret_port"] + port["tc_ret"].fillna(0.0)
        port.drop(columns=["tc_ret"], inplace=True)

    return port.set_index("date")["ret_port"].astype(float)

# =========================
# 8) Performance math
# =========================
def _infer_periods_per_year(dates: pd.Series) -> int:
    d = pd.to_datetime(dates).sort_values().dropna()
    if len(d) < 40:
        return 252
    med_step = (d.diff().dropna().dt.days.median())
    if med_step is None or med_step <= 0:
        return 252
    if med_step >= 18:   # monthly-ish
        return 12
    if med_step >= 3:    # weekly-ish
        return 52
    return 252           # daily

def perf_summary(ret: pd.Series, periods_per_year: int | None = None) -> dict:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if r.empty:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index.to_series())
    mu_d = r.mean()
    sd_d = r.std(ddof=1)
    sharpe = (mu_d / sd_d) * np.sqrt(periods_per_year) if sd_d and np.isfinite(sd_d) else np.nan
    n = len(r)
    gross = (1.0 + r).prod()
    ann_ret = gross**(periods_per_year / n) - 1.0
    ann_vol = sd_d * np.sqrt(periods_per_year)
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = (wealth / peak - 1.0).min()
    return {"ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_dd": float(dd)}

# =========================
# 9) Plotting helpers (subset)
# =========================
def plot_equity_curve(ret_s: pd.Series, title: str, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if ret_s is None or ret_s.empty: return
    ret_df = ret_s.sort_index().to_frame("ret")
    ret_df["equity"] = (1.0 + ret_df["ret"].fillna(0)).cumprod()
    plt.figure(figsize=(10,4))
    plt.plot(ret_df.index, ret_df["equity"])
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Equity"); plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_drawdown(ret_s: pd.Series, title: str, outpath: Path):
    if ret_s is None or ret_s.empty: return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    w = (1.0 + ret_s.dropna()).cumprod()
    peak = w.cummax()
    dd = w/peak - 1.0
    plt.figure(figsize=(10,3.5))
    plt.fill_between(dd.index, dd.values, 0.0, step="pre")
    plt.title(title + " — Drawdown"); plt.xlabel("Date"); plt.ylabel("Drawdown")
    _fmt_percent(plt.gca()); plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_ret_hist(ret_s: pd.Series, title: str, outpath: Path):
    if ret_s is None or ret_s.empty: return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.hist(ret_s.dropna(), bins=50)
    plt.title(title); plt.xlabel("Daily return"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# =========================
# 10) CV helpers (blocked, out-of-time)
# =========================
def build_time_blocks(months: List[pd.Period],
                      train_months: int = 36,
                      test_months: int = 6,
                      gap_months: int = 1,
                      min_train_months: int = 24) -> List[Dict[str, pd.Timestamp]]:
    """Create walk-forward time blocks (train, gap, test)."""
    months = sorted(months)
    if len(months) < (train_months + gap_months + 1):
        return []
    blocks = []
    i = train_months
    while True:
        train_start = months[i - train_months]
        train_end   = months[i - 1]
        gap_start   = months[i] if gap_months > 0 else None
        test_start_i = i + gap_months
        test_end_i   = test_start_i + test_months - 1
        if test_end_i >= len(months):
            break
        test_start = months[test_start_i]
        test_end   = months[test_end_i]
        if (i - train_months) < 0 or (test_end_i >= len(months)):
            break
        if train_months >= min_train_months:
            blocks.append(dict(
                train_start=train_start.to_timestamp("M"),
                train_end=train_end.to_timestamp("M"),
                test_start=test_start.to_timestamp("M"),
                test_end=test_end.to_timestamp("M"),
            ))
        i += test_months  # step by one test block
    return blocks

def mask_by_month_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    m = _month_eom(df["date"])
    return (m >= start) & (m <= end)

# =========================
# 11) Shock diagnostic
# =========================
def shock_sensitivity_report(df: pd.DataFrame, outdir: Path, z_thresh: float = 3.0, ret_sigma: float = 3.0) -> Path:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    if "volume" in x.columns:
        x["vol_z"] = (x.groupby("ticker")["volume"]
                        .transform(lambda s: (s - s.rolling(60, min_periods=40).mean())
                                              / (s.rolling(60, min_periods=40).std() + 1e-9)))
    else:
        x["vol_z"] = np.nan
    base_col = "ret_1d_eur" if "ret_1d_eur" in x.columns else "ret_effective"
    x["ret_z"] = (x.groupby("ticker")[base_col]
                    .transform(lambda s: (s - s.rolling(60, min_periods=40).mean())
                                          / (s.rolling(60, min_periods=40).std() + 1e-9)))
    shock_mask = (x["vol_z"].abs() > z_thresh) | (x["ret_z"].abs() > ret_sigma)
    x["shock_day"] = shock_mask.astype(int)
    outp = outdir / "shock_tags_daily.csv"
    try:
        x[["date","ticker","vol_z","ret_z","shock_day"]].to_csv(outp, index=False)
    except Exception as e:
        print(f"[warn] shock_sensitivity_report write failed: {e}")
    return outp

# =========================
# 12) Orchestrator
# =========================
def run_alpha_from_exports(
    *,
    indir: Path = DEFAULT_INDIR,
    outdir: Path = DEFAULT_OUTDIR,
    horizons: List[int] = (1,5,21,40),
    ic_method: str = "spearman",
    neutralize_by: Optional[List[str]] = ["country","supersector","size_bucket","liq_bucket"],
    robust_z: bool = False,
    rebalance: str = "M",
    top_quantile: float = 0.2,
    tc_bps: float = 5.0,
    # Rolling IC-weights
    ic_roll_m: int = 18,
    ic_shrink: float = 0.6,
    ic_max_abs_w: float = 0.40,
    ic_min_months: int = 10,
    # CV controls
    cv_enable: bool = True,
    cv_train_months: int = 36,
    cv_test_months: int = 6,
    cv_gap_months: int = 1,
    cv_min_train_months: int = 24,
) -> Dict[str, object]:
    """Main entry point. Returns dict with summaries & key outputs."""
    ensure_outdir(outdir)
    loaded = _load_exports(indir)
    df = loaded["df"].copy()
    assert_monotone_by_ticker(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ticker","date"])

    # Optional: enforce observed-history monthly universe (approximate delistings)
    if loaded.get("universe", None) is not None and not loaded["universe"].empty:
        try:
            uni = loaded["universe"]
            m2tickers = {pd.to_datetime(p): set(t) for p, t in zip(uni["month_end"], uni["tickers"])}
            df["month_eom"] = _month_eom(df["date"])
            df = df[df.apply(lambda r: (r["ticker"] in m2tickers.get(r["month_eom"], set())), axis=1)].drop(columns=["month_eom"])
        except Exception as e:
            logging.warning("universe_monthly filter skipped: %s", e)

    # Effective daily return + embargo + forwards
    df = add_effective_return(df, prefer_col="excess_ret_1d_eur", fallback_col="ret_1d_eur", out_col="ret_effective")
    df = apply_embargo(df, embargo_days=1)
    df = add_forward_returns(df, "ret_effective", horizons)
    shock_sensitivity_report(df, outdir)

    # Factor candidate set
    if loaded["factors_list"]:
        candidates = [c for c in loaded["factors_list"] if c in df.columns]
    else:
        non_factors = {"date","date_eff","ticker","country","supersector","price_eur","turnover_eur",
                       "ret_1d_eur","excess_ret_1d_eur","ret_effective","mkt_ret","beta_252","beta_rolling",
                       "adv_eur_20","ret_stock","country_mkt_ret_1d","size_bucket","liq_bucket"}
        candidates = sorted([c for c in df.columns if (df[c].dtype.kind in "fc") and (c not in non_factors)])
    if not candidates:
        raise RuntimeError("No factor columns found. Ensure DATA_ENG exports include factor columns.")

    # Neutralize & Z per factor (adds *_z and *_neu_z) — PIT-safe (per-date)
    zcols = []
    for fac in candidates:
        z_raw, z_neu = neutralize_and_z(df, fac, by=neutralize_by or [], robust=robust_z)
        df[f"{fac}_z"] = z_raw.values
        df[f"{fac}_neu_z"] = z_neu.values
        zcols.append(f"{fac}_neu_z")

    # Choose target (use the longest horizon by default for selection)
    target_h = max(horizons)
    target = f"fwd_ret_effective_{target_h}d"

    # ===== In-sample diagnostics (optional, still useful) =====
    ic_ts = compute_ic_timeseries(df, zcols, fwd_ret_col=target, freq="M", min_names=10, method=ic_method, use_embargo_date=True)
    ic_sum = summarize_ic_with_fdr(ic_ts, zcols, freq="M", h_overlapping=target_h)
    ic_ts.to_csv(outdir / "ic_timeseries.csv", index=False)
    ic_sum.to_csv(outdir / "ic_summary.csv", index=False)

    # Factor selection (IS): q_BH <= 10% and positive mean IC; else fallback to top-5 by mean
    selected = ic_sum[(ic_sum["q_BH"] <= 0.10) & (ic_sum["mean"] > 0)].copy()
    selected_factors = selected["factor"].tolist() if not selected.empty else []
    if not selected_factors:
        logging.warning("No factors passed FDR <= 10%% and positive mean IC (IS). Proceeding with top-5 by mean IC.")
        selected = ic_sum.sort_values("mean", ascending=False).head(5)
        selected_factors = selected["factor"].tolist()

    # Build static composites IS (useful to compare vs CV)
    df["score_ew"] = df[selected_factors].mean(axis=1) if selected_factors else np.nan
    iw = ic_weights(ic_sum.loc[ic_sum["factor"].isin(selected_factors), ["factor","mean"]], min_q=0.2, shrink=0.5) if selected_factors else pd.Series(dtype=float)
    iw = iw.reindex(selected_factors).fillna(0.0) if len(selected_factors) else iw
    pd.DataFrame({"weight": iw}).to_csv(outdir / "weights_ic_IS.csv")
    df["score_icw"] = df[selected_factors].mul(iw, axis=1).sum(axis=1) if selected_factors else np.nan
    rw = datewise_ridge_weights(df, selected_factors, target, alpha=2.0, min_n=15) if selected_factors else pd.Series(dtype=float)
    rw = rw.reindex(selected_factors).fillna(0.0) if len(selected_factors) else rw
    pd.DataFrame({"weight": rw}).to_csv(outdir / "weights_ridge_IS.csv")
    df["score_ridge"] = df[selected_factors].mul(rw, axis=1).sum(axis=1) if selected_factors else np.nan

    w_roll = rolling_ic_weights(ic_ts, selected_factors, lookback_months=ic_roll_m, shrink_to_ew=ic_shrink,
                                max_abs_w=ic_max_abs_w, min_months=ic_min_months, freeze_when_thin=True) if selected_factors else pd.DataFrame()
    w_roll.to_csv(outdir / "weights_ic_rolling_IS.csv")
    df["month_eom"] = _month_eom(df["date"])
    df["score_icw_roll"] = np.nan
    if not w_roll.empty:
        for m, wrow in w_roll.iterrows():
            mask = df["month_eom"] == pd.to_datetime(m)
            if mask.any():
                df.loc[mask, "score_icw_roll"] = df.loc[mask, selected_factors].mul(wrow.values, axis=1).sum(axis=1)

    # ===== Portfolios (IS diagnostic only) =====
    RET_COL = "ret_effective"
    for sname in ["score_ew","score_icw","score_ridge","score_icw_roll"]:
        if sname not in df.columns: 
            continue
        w_lo = build_longonly_weights(df, sname, topq=top_quantile, rebalance=rebalance)
        r_lo = realize_portfolio_returns(df, w_lo, ret_col=RET_COL, tc_bps=tc_bps)
        r_lo.to_csv(outdir / f"daily_returns_IS_{sname}_longonly.csv")
        plot_equity_curve(r_lo, f"(IS) {sname} — Long-only (reb={rebalance}, topq={top_quantile}, tc={tc_bps}bps)",
                          outdir / f"equity_IS_{sname}_longonly.png")
        plot_drawdown(r_lo, f"(IS) {sname} long-only", outdir / f"drawdown_IS_{sname}_longonly.png")
        plot_ret_hist(r_lo, f"(IS) {sname} daily returns (long-only)", outdir / f"ret_hist_IS_{sname}_longonly.png")

    # ===== Time-based Cross-Validation (OOS) =====
    oos_results = {}
    if cv_enable:
        months = sorted(_month_eom(df["date"]).dt.to_period("M").unique())
        blocks = build_time_blocks(months, train_months=cv_train_months, test_months=cv_test_months,
                                   gap_months=cv_gap_months, min_train_months=cv_min_train_months)
        if not blocks:
            logging.warning("No CV blocks could be formed with the given parameters.")
        else:
            logging.info("Formed %d CV blocks.", len(blocks))
            # Containers for OOS daily returns
            oos_daily = {k: [] for k in ["score_ew","score_icw","score_ridge","score_icw_roll"]}

            # Per-fold summaries
            fold_rows = []

            for i, blk in enumerate(blocks, 1):
                tr_mask = mask_by_month_range(df, blk["train_start"], blk["train_end"])
                te_mask = mask_by_month_range(df, blk["test_start"], blk["test_end"])

                df_tr = df.loc[tr_mask].copy()
                df_te = df.loc[te_mask].copy()
                if df_tr.empty or df_te.empty:
                    continue

                # IC on training only
                ic_tr = compute_ic_timeseries(df_tr, zcols, fwd_ret_col=target, freq="M",
                                              min_names=10, method=ic_method, use_embargo_date=True)
                ic_tr_sum = summarize_ic_with_fdr(ic_tr, zcols, freq="M", h_overlapping=target_h)
                # Select on training only
                sel_tr = ic_tr_sum[(ic_tr_sum["q_BH"] <= 0.10) & (ic_tr_sum["mean"] > 0)]
                sel_list = sel_tr["factor"].tolist() if not sel_tr.empty else ic_tr_sum.sort_values("mean", ascending=False).head(5)["factor"].tolist()

                # Weights from training
                # EW
                df_te["score_ew"] = df_te[sel_list].mean(axis=1) if sel_list else np.nan
                # Static IC-weighted from training
                iw_tr = ic_weights(ic_tr_sum.loc[ic_tr_sum["factor"].isin(sel_list), ["factor","mean"]], min_q=0.2, shrink=0.5) if sel_list else pd.Series(dtype=float)
                iw_tr = iw_tr.reindex(sel_list).fillna(0.0) if len(sel_list) else iw_tr
                df_te["score_icw"] = df_te[sel_list].mul(iw_tr, axis=1).sum(axis=1) if sel_list else np.nan
                # Ridge using training
                rw_tr = datewise_ridge_weights(df_tr, sel_list, target, alpha=2.0, min_n=15) if sel_list else pd.Series(dtype=float)
                rw_tr = rw_tr.reindex(sel_list).fillna(0.0) if len(sel_list) else rw_tr
                df_te["score_ridge"] = df_te[sel_list].mul(rw_tr, axis=1).sum(axis=1) if sel_list else np.nan
                # Rolling IC-weights based on training window only; freeze last train month into test
                w_roll_tr = rolling_ic_weights(ic_tr, sel_list, lookback_months=ic_roll_m, shrink_to_ew=ic_shrink,
                                               max_abs_w=ic_max_abs_w, min_months=ic_min_months, freeze_when_thin=True) if sel_list else pd.DataFrame()
                if not w_roll_tr.empty:
                    last_train_month = w_roll_tr.index.max()
                    w_last = w_roll_tr.loc[last_train_month]
                    df_te["score_icw_roll"] = df_te[sel_list].mul(w_last.values, axis=1).sum(axis=1)
                else:
                    df_te["score_icw_roll"] = np.nan

                # Portfolios on TEST only
                for sname in ["score_ew","score_icw","score_ridge","score_icw_roll"]:
                    w_lo = build_longonly_weights(df_te, sname, topq=top_quantile, rebalance=rebalance)
                    r_lo = realize_portfolio_returns(df_te, w_lo, ret_col="ret_effective", tc_bps=tc_bps)
                    if not r_lo.empty:
                        r_lo.name = f"fold{i}_{sname}"
                        oos_daily[sname].append(r_lo)
                        summ = perf_summary(r_lo)
                        fold_rows.append({
                            "fold": i, "s": sname,
                            "train_start": blk["train_start"].date(), "train_end": blk["train_end"].date(),
                            "test_start": blk["test_start"].date(),   "test_end": blk["test_end"].date(),
                            **summ
                        })

            # Aggregate OOS
            for sname, series_list in oos_daily.items():
                if series_list:
                    oos_all = pd.concat(series_list).sort_index()
                    oos_all.to_csv(outdir / f"cv_oos_daily_{sname}_longonly.csv")
                    plot_equity_curve(oos_all, f"(OOS CV) {sname} — Long-only", outdir / f"cv_equity_{sname}_longonly.png")
                    plot_drawdown(oos_all, f"(OOS CV) {sname} long-only", outdir / f"cv_drawdown_{sname}_longonly.png")
                    plot_ret_hist(oos_all, f"(OOS CV) {sname} daily returns", outdir / f"cv_ret_hist_{sname}_longonly.png")
                    oos_results[sname] = oos_all

            if fold_rows:
                cv_summary = pd.DataFrame(fold_rows)
                cv_summary.to_csv(outdir / "cv_oos_summary_by_fold.csv", index=False)
                # Overall OOS summary (equally-weight folds in time)
                overall_rows = []
                for sname in ["score_ew","score_icw","score_ridge","score_icw_roll"]:
                    if sname in oos_results:
                        overall_rows.append({**perf_summary(oos_results[sname]), "strategy": sname})
                if overall_rows:
                    pd.DataFrame(overall_rows)[["strategy","ann_ret","ann_vol","sharpe","max_dd"]].to_csv(outdir / "cv_oos_summary_overall.csv", index=False)

    logging.info("Saved outputs to %s", outdir.resolve())

    # Run metadata (data-light disclaimer)
    try:
        meta = {"mode": "data_light",
                "disclaimer": "No explicit delistings/CA/earnings/cap data; survivorship effects possible."}
        (outdir / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:
        print(f"[warn] writing RUN_METADATA.json failed: {e}")

    return {
        "ic_timeseries": ic_ts.head(),
        "ic_summary": ic_sum.head(),
        "selected_factors_IS": selected_factors,
        "weights_ic_IS": iw,
        "weights_ridge_IS": rw,
        "weights_ic_rolling_IS": w_roll.tail(3) if not w_roll.empty else w_roll,
    }

# =========================
# 13) (Optional) Visuals for cohort migration — omitted here to keep file compact
#     You can plug in your existing visualization utilities if desired.
# =========================

# =========================
# 14) CLI
# =========================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Factor selection & alpha research from DATA_ENG exports (PIT + OOS CV)")
    p.add_argument("--indir", type=Path, default=DEFAULT_INDIR, help="Input directory with DATA_ENG exports")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for alpha artifacts")
    p.add_argument("--horizons", type=int, nargs="*", default=[1,5,21,40], help="Forward return horizons (days)")
    p.add_argument("--ic_method", type=str, default="spearman", choices=["pearson","spearman"], help="IC method")
    p.add_argument("--neutralize", nargs="*", default=["country","supersector","size_bucket","liq_bucket"], help="Neutralize by (empty to disable)")
    p.add_argument("--robust_z", action="store_true", help="Use robust median/MAD z-scores")
    p.add_argument("--rebalance", type=str, default="M", help="Rebalance frequency: 'M' month-end, 'W' weekly, or 'D' daily")
    p.add_argument("--topq", type=float, default=0.2, help="Top-quantile for long-only tilts (e.g., 0.2)")
    p.add_argument("--tc_bps", type=float, default=5.0, help="Transaction cost per 100% turnover, in bps")
    # Rolling IC-weights
    p.add_argument("--ic_roll_m", type=int, default=18, help="Rolling months for IC-weight lookback (e.g., 12–24)")
    p.add_argument("--ic_shrink", type=float, default=0.6, help="Shrink toward equal-weight (0..1, e.g., 0.6)")
    p.add_argument("--ic_max_abs_w", type=float, default=0.40, help="Max absolute weight per factor after norm & clipping")
    p.add_argument("--ic_min_months", type=int, default=10, help="Freeze/fallback when sample < this many months")
    # CV
    p.add_argument("--cv_enable", action="store_true", help="Enable time-based cross-validation (OOS)")
    p.add_argument("--cv_train_months", type=int, default=36, help="Train window (months)")
    p.add_argument("--cv_test_months", type=int, default=6, help="Test window (months)")
    p.add_argument("--cv_gap_months", type=int, default=1, help="Gap between train and test (months)")
    p.add_argument("--cv_min_train_months", type=int, default=24, help="Minimum months required in train to form a fold")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_alpha_from_exports(
        indir=args.indir,
        outdir=args.outdir,
        horizons=args.horizons,
        ic_method=args.ic_method,
        neutralize_by=args.neutralize if args.neutralize is not None else [],
        robust_z=args.robust_z,
        rebalance=args.rebalance,
        top_quantile=args.topq,
        tc_bps=args.tc_bps,
        ic_roll_m=args.ic_roll_m,
        ic_shrink=args.ic_shrink,
        ic_max_abs_w=args.ic_max_abs_w,
        ic_min_months=args.ic_min_months,
        cv_enable=args.cv_enable,
        cv_train_months=args.cv_train_months,
        cv_test_months=args.cv_test_months,
        cv_gap_months=args.cv_gap_months,
        cv_min_train_months=args.cv_min_train_months,
    )