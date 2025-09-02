# ==========================================
# factors_module.py — Factors-only pipeline
# ==========================================
# Inputs:  Prices.csv, stoxx_europe_600_v2.csv, index_data.csv
# Outputs (under DATA_ENG/):
#   - factors_daily.csv / .parquet
#   - factors_eom_z.csv / .parquet
#   - factor_completeness.csv
#   - factor_coverage_daily.csv
#   - factor_corr_latest.csv
#   - factor_corr_eom_avg.csv
#   - factors_list.json
#   - country_market_returns.csv
#   - global_market_ew.csv
#   - universe_monthly.csv               <-- NEW (observed-history universe)
#
# Usage:
#   python factors_module.py
#
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------
# Tunables / defaults
# --------------------
OUTDIR = Path("1_0_DATA_ENG")
MIN_CROSS_SEC = 10         # min names per day to keep in global EW market
BETA_LONG_WIN = 252        # 1y trading days
BETA_LONG_MIN = 200
BETA_SHORT_WIN = 60        # ~3 months
BETA_SHORT_MIN = 40

# --------------------
# Small I/O helpers
# --------------------
def _read_csv_flex(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def _export(df: pd.DataFrame, base: Path):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
    except Exception as e:
        print(f"[warn] Parquet export skipped for {base.name}: {e}")

# --------------------
# Utilities (data-light)
# --------------------
def detect_splits_unadjusted(df: pd.DataFrame, price_col="price_eur", vol_col="volume", tol=0.02):
    """
    Heuristic split/reverse-split detector for unadjusted price sources.
    Logs candidates; does not modify prices.
    Flags price ratios near {2,3,5,10} (or inverse) with a same-day volume blip.
    """
    out = []
    if price_col not in df.columns:
        return out
    for tic, g in df[["date", price_col, vol_col]].dropna().sort_values(["date"]).groupby(df["ticker"]):
        p = g[price_col].to_numpy(dtype=float)
        if p.size < 2:
            continue
        v = g[vol_col].to_numpy(dtype=float) if vol_col in g.columns else np.full_like(p, np.nan)
        r = np.divide(p[1:], p[:-1], where=(p[:-1] != 0))
        targets = [2, 3, 5, 10]
        mask = np.zeros_like(r, dtype=bool)
        for k in targets:
            mask |= np.isclose(r, 1.0/k, atol=tol) | np.isclose(r, k, atol=tol)
        # crude volume blip (doubling)
        vol_blip = (np.abs(pd.Series(v).pct_change().fillna(0.0).to_numpy()[1:]) > 1.0)
        idx = np.where(mask & vol_blip)[0]
        for i in idx:
            out.append((str(tic), pd.to_datetime(g.iloc[i+1]["date"]).date().isoformat(), float(r[i])))
    if out:
        print(f"[warn] Potential splits detected (first 5): {out[:5]} ... total={len(out)}")
    return out

def build_universe_monthly(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Observed-history investable universe per month (data-light delisting proxy).
    """
    x = model_df[["date","ticker"]].dropna().copy()
    x["month"] = pd.to_datetime(x["date"]).dt.to_period("M")
    uni = (x.drop_duplicates(["month","ticker"])
             .groupby("month")["ticker"].apply(list)
             .reset_index(name="tickers"))
    uni["month_end"] = uni["month"].dt.to_timestamp("M")
    # store as JSON strings to avoid eval downstream
    uni["tickers"] = uni["tickers"].apply(lambda lst: json.dumps(lst))
    return uni[["month","month_end","tickers"]]

# --------------------
# Core factor builder
# --------------------
def build_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: Date, Close, Volume, Ticker_YF, Supersector, Country
    Returns tidy daily panel with:
      ret_1d_eur, mom_12_1, mom_6_1, rev_5d, hi52_prox,
      vol_60d, vol_252d, adv_eur_20, size_proxy, amihud_20d, zero_ret_20d, max5_21d,
      country_rel_mom_6_1, sector_rel_mom_6_1
      + (NEW) liq_proxy, size_bucket, liq_bucket
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close", "Ticker_YF"]).rename(
        columns={
            "Date":"date",
            "Ticker_YF":"ticker",
            "Supersector":"supersector",
            "Country":"country",
            "Close":"price_eur",
            "Volume":"volume",
        }
    )
    df = (df[["date","ticker","price_eur","volume","supersector","country"]]
            .drop_duplicates(subset=["date","ticker"])
            .sort_values(["ticker","date"]))

    # --- Split sanity check before returns (logs only; no price adjustment here)
    try:
        detect_splits_unadjusted(df, price_col="price_eur", vol_col="volume", tol=0.02)
    except Exception as _e:
        print(f"[warn] split check skipped: {_e}")

    # 1) daily returns
    df["ret_1d_eur"] = df.groupby("ticker")["price_eur"].pct_change()

    # 2) momentum (12–1)
    def _mom_12_1(s: pd.Series) -> pd.Series:
        cs = (1 + s.fillna(0)).cumprod()
        return cs.shift(21) / cs.shift(252 + 21) - 1.0
    df["mom_12_1"] = df.groupby("ticker")["ret_1d_eur"].transform(_mom_12_1)

    # 3) rolling vol
    df["vol_60d"]  = (df.groupby("ticker")["ret_1d_eur"]
                        .rolling(60,  min_periods=40).std()
                        .reset_index(level=0, drop=True))
    df["vol_252d"] = (df.groupby("ticker")["ret_1d_eur"]
                        .rolling(252, min_periods=200).std()
                        .reset_index(level=0, drop=True))

    # 4) size/liquidity primitives
    df["turnover_eur"] = df["price_eur"] * df["volume"].fillna(0)
    df["adv_eur_20"]   = (df.groupby("ticker")["turnover_eur"]
                            .rolling(20, min_periods=10).mean()
                            .reset_index(level=0, drop=True))
    df["size_proxy"]   = np.log(df["adv_eur_20"].replace(0, np.nan))

    # ===== additional price/volume factors =====
    # A) short-term reversal = - 5d return
    def _roll_cumret(s: pd.Series, w: int) -> pd.Series:
        return (1.0 + s).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1.0
    df["ret_5d"] = df.groupby("ticker")["ret_1d_eur"].transform(lambda s: _roll_cumret(s, 5))
    df["rev_5d"] = -df["ret_5d"]

    # B) momentum (6–1)
    def _mom_6_1(s: pd.Series) -> pd.Series:
        cs = (1 + s.fillna(0)).cumprod()
        return cs.shift(21) / cs.shift(126 + 21) - 1.0
    df["mom_6_1"] = df.groupby("ticker")["ret_1d_eur"].transform(_mom_6_1)

    # C) 52-week high proximity
    hi_252 = (df.groupby("ticker")["price_eur"]
                .rolling(252, min_periods=20).max()
                .reset_index(level=0, drop=True))
    df["hi52_prox"] = df["price_eur"] / hi_252 - 1.0

    # D) Amihud 20d
    amihud_daily = df["ret_1d_eur"].abs() / df["turnover_eur"].replace(0, np.nan)
    df["amihud_20d"] = (amihud_daily.groupby(df["ticker"])
                        .rolling(20, min_periods=10).mean()
                        .reset_index(level=0, drop=True))

    # E) Zero-return share 20d
    zero_flag = df["ret_1d_eur"].fillna(0).eq(0.0).astype(float)
    df["zero_ret_20d"] = (zero_flag.groupby(df["ticker"])
                          .rolling(20, min_periods=10).mean()
                          .reset_index(level=0, drop=True))

    # F) MAX effect: mean of 5 largest daily returns over 21d
    def _mean_top_k(arr, k=5):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0: return np.nan
        k = min(k, a.size)
        idx = np.argpartition(a, -k)[-k:]
        return float(np.mean(a[idx]))
    df["max5_21d"] = (df.groupby("ticker")["ret_1d_eur"]
                      .rolling(21, min_periods=10)
                      .apply(lambda x: _mean_top_k(x, k=5), raw=True)
                      .reset_index(level=0, drop=True))

    # Relative-strength vs country / sector (based on mom_6_1)
    by_date_country   = df.groupby(["date","country"])["mom_6_1"].transform("median")
    by_date_supersect = df.groupby(["date","supersector"])["mom_6_1"].transform("median")
    df["country_rel_mom_6_1"] = df["mom_6_1"] - by_date_country
    df["sector_rel_mom_6_1"]  = df["mom_6_1"] - by_date_supersect

    # ---- NEW: Liquidity proxy & coarse size/liquidity buckets ----
    df["liq_proxy"] = -np.log(df["amihud_20d"].replace(0, np.nan))

    def _bucket_q(s: pd.Series, qs=(0.2,0.4,0.6,0.8)) -> pd.Series:
        try:
            return pd.qcut(s, q=list(qs)+[1.0], labels=False, duplicates="drop")
        except Exception:
            # low cross-section → return NaNs to avoid breaking neutralization
            return pd.Series(index=s.index, dtype="float64")

    df["size_bucket"] = df.groupby("date")["size_proxy"].transform(_bucket_q)
    df["liq_bucket"]  = df.groupby("date")["liq_proxy"].transform(_bucket_q)

    cols = [
        "date","ticker","country","supersector",
        "price_eur","ret_1d_eur",
        "mom_12_1","mom_6_1","rev_5d","hi52_prox",
        "vol_60d","vol_252d",
        "adv_eur_20","size_proxy","liq_proxy","size_bucket","liq_bucket",
        "amihud_20d","zero_ret_20d","max5_21d",
        "country_rel_mom_6_1","sector_rel_mom_6_1",
    ]
    return df[cols].dropna(subset=["ret_1d_eur"])

# ----------------------------
# Country market & excess ret
# ----------------------------
def prepare_country_index_returns(index_df: pd.DataFrame,
                                  country_map: dict | None = None) -> pd.DataFrame:
    """
    Standardize index_data and compute daily country market returns.
    Accepts columns (any case): Date/date, Close/close, Ticker_YF/ticker, Country/country
    """
    idx = index_df.copy()

    # normalize
    if "date" in idx.columns: idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
    else:                     idx["date"] = pd.to_datetime(idx["Date"], errors="coerce")
    if "ticker" in idx.columns: idx["index_ticker"] = idx["ticker"].astype(str)
    else:                        idx["index_ticker"] = idx["Ticker_YF"].astype(str)
    if "country" in idx.columns: idx["country"] = idx["country"].astype(str)
    else:                        idx["country"] = idx["Country"].astype(str)
    if "close" in idx.columns: idx["index_close"] = pd.to_numeric(idx["close"], errors="coerce")
    else:                       idx["index_close"] = pd.to_numeric(idx["Close"], errors="coerce")

    if country_map:
        idx["country"] = idx["country"].map(lambda x: country_map.get(x, x))

    idx = idx.dropna(subset=["date","index_close","index_ticker","country"]).sort_values(["index_ticker","date"])

    idx["idx_ret_1d"] = idx.groupby("index_ticker")["index_close"].pct_change()

    country_ret = (idx.groupby(["country","date"])["idx_ret_1d"]
                     .mean()
                     .rename("country_mkt_ret_1d")
                     .reset_index())
    return country_ret

def attach_country_returns(stock_df: pd.DataFrame, country_ret: pd.DataFrame) -> pd.DataFrame:
    out = (stock_df.merge(country_ret, on=["date","country"], how="left")
                   .sort_values(["ticker","date"]))
    out["excess_ret_1d_eur"] = out["ret_1d_eur"] - out["country_mkt_ret_1d"]
    return out

# ----------------------------
# Market series & beta factors
# ----------------------------
def _pick_return_col(df: pd.DataFrame, prefer_excess: bool = False) -> str:
    if prefer_excess and "excess_ret_1d_eur" in df.columns:
        return "excess_ret_1d_eur"
    if "ret_1d_eur" in df.columns:
        return "ret_1d_eur"
    raise KeyError("Need 'excess_ret_1d_eur' or 'ret_1d_eur' in df.")

def build_global_market(df: pd.DataFrame,
                        min_cross_sec: int = MIN_CROSS_SEC,
                        prefer_excess: bool = False) -> pd.DataFrame:
    ret_col = _pick_return_col(df, prefer_excess=prefer_excess)
    xsec = (df.groupby("date")
              .agg(n=("ticker","nunique"), mkt_ret=(ret_col,"mean"))
              .reset_index())
    xsec = xsec.loc[xsec["n"] >= min_cross_sec, ["date","mkt_ret"]]
    return xsec

def attach_market_and_betas(df: pd.DataFrame,
                            mkt: pd.DataFrame | None = None,
                            use_country_market_if_present: bool = True) -> pd.DataFrame:
    """
    Adds:
      - mkt_ret (country or global EW)
      - beta_252 (rolling vs mkt)
      - beta_60  (rolling vs mkt)
    """
    df = df.copy().sort_values(["ticker","date"])
    if use_country_market_if_present and "country_mkt_ret_1d" in df.columns and df["country_mkt_ret_1d"].notna().any():
        df["mkt_ret"] = pd.to_numeric(df["country_mkt_ret_1d"], errors="coerce")
    else:
        if mkt is None:
            mkt = build_global_market(df, min_cross_sec=MIN_CROSS_SEC, prefer_excess=False)
        df = df.merge(mkt, on="date", how="inner")  # restrict to mkt dates

    # choose return for beta (raw if available)
    ret_for_beta = "ret_1d_eur" if "ret_1d_eur" in df.columns else _pick_return_col(df, prefer_excess=False)

    # rolling betas
    pieces_long, pieces_short = [], []
    for t, g in df.groupby("ticker", sort=False):
        cov_L = g[ret_for_beta].rolling(BETA_LONG_WIN, min_periods=BETA_LONG_MIN).cov(g["mkt_ret"])
        var_L = g["mkt_ret"].rolling(BETA_LONG_WIN, min_periods=BETA_LONG_MIN).var()
        beta_L = cov_L / var_L

        cov_S = g[ret_for_beta].rolling(BETA_SHORT_WIN, min_periods=BETA_SHORT_MIN).cov(g["mkt_ret"])
        var_S = g["mkt_ret"].rolling(BETA_SHORT_WIN, min_periods=BETA_SHORT_MIN).var()
        beta_S = cov_S / var_S

        tmpL = g[["date","ticker"]].copy(); tmpL["beta_252"] = beta_L.values; pieces_long.append(tmpL)
        tmpS = g[["date","ticker"]].copy(); tmpS["beta_60"]  = beta_S.values; pieces_short.append(tmpS)

    df = df.merge(pd.concat(pieces_long, ignore_index=True), on=["date","ticker"], how="left")
    df = df.merge(pd.concat(pieces_short, ignore_index=True), on=["date","ticker"], how="left")
    return df

# ----------------------------
# X-sec helpers (EoM & z-scores)
# ----------------------------
def last_business_day_flags(dates: pd.Series) -> pd.Series:
    m = dates.dt.to_period("M")
    return dates.groupby(m).transform("max").eq(dates)

def xsec_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0: return s * 0.0
    return (s - s.mean()) / std

def _neutralize_xsec(rebal: pd.DataFrame, factor_col: str) -> pd.Series:
    """
    Residualize factor by country + supersector (+ size_bucket + liq_bucket if present) per date using OLS with dummies.
    """
    out = pd.Series(index=rebal.index, dtype="float64")
    base_cols = ["country","supersector"]
    opt_cols  = [c for c in ["size_bucket","liq_bucket"] if c in rebal.columns]
    tmp = rebal[["date"] + base_cols + opt_cols + [factor_col]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp[factor_col] = pd.to_numeric(tmp[factor_col], errors="coerce")
    for c in base_cols + opt_cols:
        if c in tmp.columns:
            tmp[c] = tmp[c].astype("category")

    for d, g in tmp.groupby("date", sort=False):
        y = g[factor_col].to_numpy(dtype="float64")
        valid = np.isfinite(y)
        if valid.sum() < 2:
            out.loc[g.index] = np.nan
            continue
        g = g.loc[valid]; y = y[valid]
        cols = [c for c in base_cols + opt_cols if c in g.columns]
        Xd = pd.get_dummies(g[cols], drop_first=True, dtype=float) if cols else pd.DataFrame(index=g.index)
        if Xd.shape[1] == 0:
            resid = y - np.nanmean(y)
        else:
            X = np.column_stack([np.ones(len(g), dtype="float64"), Xd.to_numpy(dtype="float64", na_value=0.0)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
        out.loc[g.index] = resid

    return out

def make_eom_zscores(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """
    Keeps last calendar date per month present in df and returns
    z-scored + neutralized z-scored factors per EoM.
    Neutralization uses country+supersector and (if present) size_bucket/liq_bucket.
    """
    e = df.copy()
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e["is_eom"] = last_business_day_flags(e["date"])
    e = e[e["is_eom"]].copy()
    for f in factor_cols:
        e[f] = pd.to_numeric(e[f], errors="coerce")
        e[f"{f}_z"] = e.groupby("date")[f].transform(xsec_zscore)
        neu = _neutralize_xsec(e[["date","country","supersector","size_bucket","liq_bucket",f]].copy(), f)
        e[f"{f}_neu"] = neu
        e[f"{f}_neu_z"] = e.groupby("date")[f"{f}_neu"].transform(xsec_zscore)
    keep = ["date","ticker","country","supersector","size_bucket","liq_bucket"] + [c for c in e.columns if c.endswith("_z")]
    return e[keep].sort_values(["date","ticker"])

# ----------------------------
# Completeness & correlations
# ----------------------------
def factor_columns_from(df: pd.DataFrame) -> list[str]:
    non_factors = {
        "date","ticker","country","supersector","price_eur",
        "turnover_eur","ret_5d","ret_1d_eur","excess_ret_1d_eur",
        "country_mkt_ret_1d","mkt_ret",
        # new helper/non-signal fields we DON'T want treated as factors:
        "size_bucket","liq_bucket"
    }
    return [c for c in df.columns if c not in non_factors and df[c].dtype.kind in "fc"]

def completeness_table(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    total = len(df)
    rows = []
    for f in factor_cols:
        nonnull = df[f].notna().sum()
        pct = nonnull / total if total else np.nan
        dates_with = df.loc[df[f].notna(),"date"]
        start = pd.to_datetime(dates_with.min()).date() if not dates_with.empty else pd.NaT
        end   = pd.to_datetime(dates_with.max()).date() if not dates_with.empty else pd.NaT
        avg_n = (df.groupby("date")[f].apply(lambda s: s.notna().sum()).mean()
                 if not df.empty else np.nan)
        rows.append({"factor": f, "non_null": int(nonnull), "total_rows": int(total),
                     "pct_non_null": pct, "avg_names_per_date": avg_n,
                     "first_date": start, "last_date": end})
    return pd.DataFrame(rows).sort_values("factor")

def daily_coverage(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    # per-date non-null counts for each factor
    agg = df.groupby("date")[factor_cols].apply(lambda g: g.notna().sum()).reset_index()
    return agg.sort_values("date")

def corr_latest_eom(eom_z: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    last_dt = eom_z["date"].max()
    snap = eom_z[eom_z["date"]==last_dt][[f"{f}_z" for f in factor_cols if f"{f}_z" in eom_z.columns]].dropna()
    return snap.corr(method="pearson")

def corr_avg_over_eoms(eom_z: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    zs = [f"{f}_z" for f in factor_cols if f"{f}_z" in eom_z.columns]
    dates = sorted(eom_z["date"].dropna().unique())
    acc = None; k = 0
    for d in dates:
        M = eom_z[eom_z["date"]==d][zs].dropna()
        if M.shape[0] < 3:  # skip tiny cross-sections
            continue
        C = M.corr(method="pearson")
        if acc is None:
            acc = C
        else:
            acc = acc.add(C, fill_value=0.0)
        k += 1
    if acc is None or k == 0:
        return pd.DataFrame()
    return acc / k

# ----------------------------
# Driver
# ----------------------------
def _export_compat_frames(
    *,
    outdir: Path,
    model_df_v1: pd.DataFrame,
    model_df_v2: pd.DataFrame,
    market_series: pd.DataFrame,
    df_with_beta: pd.DataFrame,
    corr_latest: pd.DataFrame,
    last_dt: pd.Timestamp
) -> None:
    """Write legacy-compatible artifacts that downstream .py files read."""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) model_df_v1 — pre-index-attach factors panel
    _export(
        model_df_v1.sort_values(["ticker","date"]),
        outdir / "model_df_v1"
    )

    # 2) model_df_v2 — after country attach (has country_mkt_ret_1d & excess_ret_1d_eur)
    _export(
        model_df_v2.sort_values(["ticker","date"]),
        outdir / "model_df_v2"
    )

    # 3) market_series — ['date','mkt_ret']
    market_series.sort_values("date").to_csv(outdir / "market_series.csv", index=False)
    market_series.sort_values("date").to_csv(outdir / "global_market_ew.csv", index=False)   # alias (per header)

    # 4) df_with_beta — after attaching market + betas
    _export(
        df_with_beta.sort_values(["ticker","date"]),
        outdir / "df_with_beta"
    )

    # 5) corr_latest — legacy filename + header-consistent alias
    corr_latest.to_csv(outdir / "cross_section_correlations_latest.csv")
    corr_latest.to_csv(outdir / "factor_corr_latest.csv")

    # 6) last_date — simple ISO text for downstream loaders
    with (outdir / "last_date.txt").open("w", encoding="utf-8") as f:
        f.write(pd.to_datetime(last_dt).date().isoformat())

def run_pipeline(
    prices_csv="Prices.csv",
    crosswalk_csv="stoxx_europe_600_v2.csv",
    index_csv="index_data.csv",
    *,
    outdir: Path = OUTDIR,
    min_cross_sec: int = MIN_CROSS_SEC
) -> dict:
    """
    End-to-end build:
      - model_df_v1: daily factors
      - model_df_v2: + country market & excess returns
      - market_series: global EW market (or set min_cross_sec to 8 to mirror your older run)
      - df_with_beta: + betas (beta_252, beta_60)
      - universe_monthly: observed-history investable list (data-light delisting proxy)
    Also writes legacy-compatible CSVs and returns the legacy dict keys.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Load & join
    prices_df = _read_csv_flex(prices_csv).rename(columns={"ticker":"Ticker_YF"})
    xwalk     = _read_csv_flex(crosswalk_csv)
    index_df  = _read_csv_flex(index_csv).rename(columns={"ticker":"Ticker_YF"})

    stocks = prices_df.merge(
        xwalk[["Supersector","Country","Ticker_YF"]],
        how="left",
        on="Ticker_YF"
    )

    # 1) Daily factors (legacy: model_df_v1)
    model_df_v1 = build_model_df(stocks)

    # 1b) Universe per month (data-light delisting proxy)
    try:
        uni = build_universe_monthly(model_df_v1)
        uni.to_csv(outdir / "universe_monthly.csv", index=False)
    except Exception as e:
        print(f"[warn] universe_monthly export failed: {e}")

    # 2) Country market ret + excess (legacy: model_df_v2)
    country_ret = prepare_country_index_returns(index_df)
    (outdir / "country_market_returns.csv").write_text(country_ret.to_csv(index=False))
    model_df_v2 = attach_country_returns(model_df_v1, country_ret)

    # 3) Global EW market (legacy: market_series)
    market_series = build_global_market(model_df_v2, min_cross_sec=min_cross_sec, prefer_excess=False)

    # 4) Betas (legacy: df_with_beta) — includes beta_252 (and beta_60)
    df_with_beta = attach_market_and_betas(model_df_v2, mkt=market_series, use_country_market_if_present=True)

    # ret_stock (legacy convention: prefer excess if present)
    if "excess_ret_1d_eur" in df_with_beta.columns:
        df_with_beta["ret_stock"] = pd.to_numeric(df_with_beta["excess_ret_1d_eur"], errors="coerce")
    else:
        df_with_beta["ret_stock"] = pd.to_numeric(df_with_beta["ret_1d_eur"], errors="coerce")

    # 5) EoM z-scores, completeness, coverage, average correlations (new richer artifacts)
    factor_cols = sorted(factor_columns_from(df_with_beta))
    eom_z       = make_eom_zscores(df_with_beta, factor_cols)
    comp        = completeness_table(df_with_beta, factor_cols)
    cov         = daily_coverage(df_with_beta, factor_cols)

    # Legacy corr (latest EoM or latest cross-section if EoM empty)
    if not eom_z.empty:
        corr_latest_legacy = corr_latest_eom(eom_z, factor_cols)
        last_dt = eom_z["date"].max()
    else:
        # Fallback: compute from latest date present (keeps legacy contract)
        last_dt = pd.to_datetime(df_with_beta["date"]).max()
        cols = [c for c in ["beta_252","mom_12_1","vol_252d","size_proxy","ret_stock"] if c in df_with_beta.columns]
        snap = df_with_beta[df_with_beta["date"]==last_dt][cols].dropna()
        corr_latest_legacy = snap.corr(method="pearson") if not snap.empty else pd.DataFrame()

    # Also compute EoM-avg correlation (new)
    corr_eom_avg = corr_avg_over_eoms(eom_z, factor_cols)

    # Legacy-compatible exports
    _export_compat_frames(
        outdir=outdir,
        model_df_v1=model_df_v1,
        model_df_v2=model_df_v2,
        market_series=market_series,
        df_with_beta=df_with_beta,
        corr_latest=corr_latest_legacy,
        last_dt=last_dt,
    )

    # Keep richer exports you already liked
    _export(
        df_with_beta[["date","ticker","country","supersector","size_bucket","liq_bucket"] + factor_cols + ["ret_1d_eur","excess_ret_1d_eur","mkt_ret","beta_252","beta_60","ret_stock"]]
        .sort_values(["ticker","date"]),
        outdir / "factors_daily"
    )
    _export(eom_z, outdir / "factors_eom_z")
    comp.to_csv(outdir / "factor_completeness.csv", index=False)
    cov.to_csv(outdir / "factor_coverage_daily.csv", index=False)
    corr_eom_avg.to_csv(outdir / "factor_corr_eom_avg.csv")

    # Convenience: keep the top/bottom beta snapshots (unchanged filenames)
    snap = df_with_beta.loc[df_with_beta["date"]==last_dt,
                            ["ticker","country","supersector","beta_252","mom_12_1","vol_252d","size_proxy","ret_stock"]]
    snap.sort_values("beta_252", ascending=False).head(20).to_csv(outdir/"top20_beta.csv", index=False)
    snap.sort_values("beta_252", ascending=True ).head(20).to_csv(outdir/"bottom20_beta.csv", index=False)

    # Factors list JSON (for downstream)
    try:
        (outdir / "factors_list.json").write_text(json.dumps(factor_cols, indent=2))
    except Exception as e:
        print(f"[warn] factors_list.json export failed: {e}")

    print(f"[ok] Wrote legacy & new outputs to {outdir.resolve()}")

    # === Return BOTH legacy and new keys ===
    return {
        # legacy keys required by downstream
        "model_df_v1": model_df_v1,
        "model_df_v2": model_df_v2,
        "market_series": market_series,
        "df_with_beta": df_with_beta,
        "corr_latest": corr_latest_legacy,
        "last_date": last_dt,

        # richer objects (kept)
        "factors_daily": df_with_beta,
        "factors_eom_z": eom_z,
        "factor_list": factor_cols,
        "factor_completeness": comp,
        "factor_coverage_daily": cov,
        "corr_eom_avg": corr_eom_avg,
        "country_market": country_ret,
        "global_market": market_series,
        "universe_monthly": uni if 'uni' in locals() else None,
    }

# -------------
# CLI entrypoint
# -------------
if __name__ == "__main__":
    run_pipeline()
