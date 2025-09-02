#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_pipeline.py — Full EDA using outputs from DATA_ENG (factors_module) step.

Reads the exports written by the previous step (e.g., DATA_ENG/model_df_v1.csv,
DATA_ENG/model_df_v2.csv, DATA_ENG/market_series.csv, DATA_ENG/df_with_beta.csv,
DATA_ENG/factors_daily.parquet, DATA_ENG/factors_eom_z.parquet, etc.),
then produces charts, correlation tables, and an HTML index.

Usage examples:
  python eda_pipeline.py
  python eda_pipeline.py --indir DATA_ENG --outdir eda_outputs --winsorize --neutralize country supersector
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================
# 0) Logging & constants
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
plt.rcParams["figure.dpi"] = 120

# Defaults
DEFAULT_INDIR  = Path("1_0_DATA_ENG")
DEFAULT_OUTDIR = Path("2_0_EDA")

# =========================
# 1) Utilities & validators
# =========================
def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def _check_cols(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required columns: {missing}")

def _as_category(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out

def _downcast_float(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    out = df.copy()
    candidates = cols if cols is not None else [c for c in out.columns if pd.api.types.is_float_dtype(out[c])]
    for c in candidates:
        out[c] = pd.to_numeric(out[c], downcast="float")
    return out

def winsorize_xsec(s: pd.Series, lo_q: float = 0.01, hi_q: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(lo_q)
    hi = s.quantile(hi_q)
    return s.clip(lower=lo, upper=hi)

def zscore_xsec(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd

def summarize_nan_share_by_date(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    g = df.groupby("date")[cols]
    # fraction NA per col per date
    out = 1.0 - g.count().div(g.size(), axis=0)
    return out

def _read_best(path_base: Path) -> Optional[pd.DataFrame]:
    """Load base.(parquet|csv) if present; return None if neither exists."""
    pq = path_base.with_suffix(".parquet")
    cs = path_base.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    return None

def _load_exports(indir: Path):
    """Load DATA_ENG exports with graceful fallbacks."""
    indir = Path(indir)
    # Core frames
    df_with_beta = _read_best(indir / "df_with_beta")
    if df_with_beta is None:
        # use richer factors_daily if present
        df_with_beta = _read_best(indir / "factors_daily")
    if df_with_beta is None:
        raise FileNotFoundError("Could not find df_with_beta.(csv|parquet) or factors_daily.(csv|parquet) in DATA_ENG.")

    model_df_v1 = _read_best(indir / "model_df_v1")
    model_df_v2 = _read_best(indir / "model_df_v2")

    # Market
    market_series = _read_best(indir / "market_series")
    if market_series is None:
        market_series = _read_best(indir / "global_market_ew")

    # EOM z-scores & factor list (optional)
    factors_eom_z = _read_best(indir / "factors_eom_z")
    factors_list_path = indir / "factors_list.json"
    factors_list = None
    if factors_list_path.exists():
        try:
            factors_list = json.loads(factors_list_path.read_text(encoding="utf-8"))
        except Exception:
            factors_list = None

    # Last date helper
    last_date_txt = indir / "last_date.txt"
    if last_date_txt.exists():
        try:
            last_date = pd.to_datetime(last_date_txt.read_text().strip())
        except Exception:
            last_date = pd.to_datetime(df_with_beta["date"]).max()
    else:
        last_date = pd.to_datetime(df_with_beta["date"]).max()

    return {
        "df_with_beta": df_with_beta,
        "model_df_v1": model_df_v1,
        "model_df_v2": model_df_v2,
        "market_series": market_series,
        "factors_eom_z": factors_eom_z,
        "factors_list": factors_list,
        "last_date": last_date
    }

# =======================================
# 2) Fast rolling beta t-stat (optional)
# =======================================
def attach_beta_tstat(df: pd.DataFrame,
                      *,
                      ret_col: str = "ret_1d_eur",
                      mkt_col: str = "mkt_ret",
                      roll_win: int = 252,
                      min_periods: int = 200,
                      out_col: str = "beta_252",
                      tstat_col: str = "beta_tstat") -> pd.DataFrame:
    """Add rolling OLS t-stat for an existing rolling beta setup (vectorized)."""
    need = [ret_col, mkt_col, "ticker", "date", out_col]
    _check_cols(df, need, "attach_beta_tstat")
    df = df.copy().sort_values(["ticker","date"])

    # Moments for residual variance
    df["_prod"] = df[ret_col] * df[mkt_col]
    df["_r2"]   = df[ret_col] * df[ret_col]
    df["_m2"]   = df[mkt_col] * df[mkt_col]

    g = df.groupby("ticker", sort=False)
    roll = lambda s: s.rolling(roll_win, min_periods=min_periods)

    E_R  = g[ret_col].transform(lambda s: roll(s).mean())
    E_M  = g[mkt_col].transform(lambda s: roll(s).mean())
    E_RM = g["_prod"].transform(lambda s: roll(s).mean())
    E_R2 = g["_r2"].transform(lambda s: roll(s).mean())
    E_M2 = g["_m2"].transform(lambda s: roll(s).mean())
    N    = g[ret_col].transform(lambda s: roll(s).count())

    cov  = E_RM - (E_R * E_M)
    varM = E_M2 - (E_M * E_M)
    beta = df[out_col]

    varR = E_R2 - (E_R * E_R)
    var_resid = (varR - 2.0 * beta * cov + (beta * beta) * varM).clip(lower=0)
    denom = (N - 2) * varM
    se_b = ((N * var_resid) / denom).where(denom > 0).pow(0.5)
    tstat = beta / se_b.where(se_b != 0)

    df[tstat_col] = tstat.values
    df.drop(columns=["_prod","_r2","_m2"], inplace=True)
    return df

# ==========================
# 3) Visualization helpers
# ==========================
def _fmt_percent(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)

def plot_coverage(df: pd.DataFrame, outdir: Path):
    ensure_outdir(outdir)
    cov = df.groupby("date")["ticker"].nunique()
    plt.figure(figsize=(10,4))
    plt.plot(cov.index, cov.values)
    plt.title("Coverage — # tickers over time")
    plt.xlabel("Date"); plt.ylabel("# tickers"); plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(outdir / "coverage_count.png"); plt.close()

def plot_market(df_or_mkt: pd.DataFrame, outdir: Path):
    ensure_outdir(outdir)
    if "mkt_ret" in df_or_mkt.columns and "ticker" not in df_or_mkt.columns:
        m = df_or_mkt.drop_duplicates("date")[["date","mkt_ret"]].sort_values("date")
    elif "mkt_ret" in df_or_mkt.columns:
        m = df_or_mkt.drop_duplicates("date")[["date","mkt_ret"]].sort_values("date")
    else:
        # rebuild equal-weight market from ret columns if needed
        if "excess_ret_1d_eur" in df_or_mkt.columns:
            ret_col = "excess_ret_1d_eur"
        else:
            ret_col = "ret_1d_eur"
        g = df_or_mkt.groupby("date")[ret_col].mean().rename("mkt_ret").reset_index()
        m = g.sort_values("date")
    level = (1.0 + m["mkt_ret"].fillna(0)).cumprod()
    plt.figure(figsize=(10,4))
    plt.plot(m["date"], level)
    plt.title("Cross-Section Market Index (normalized)")
    plt.xlabel("Date"); plt.ylabel("Index level"); plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(outdir / "market_index.png"); plt.close()

def _pick_beta_col(df: pd.DataFrame) -> str:
    for c in ["beta_252", "beta_rolling", "beta"]:
        if c in df.columns:
            return c
    raise KeyError("No beta column found (looked for 'beta_252','beta_rolling','beta').")

def plot_beta_hist(df: pd.DataFrame, dt, outdir: Path):
    ensure_outdir(outdir)
    beta_col = _pick_beta_col(df)
    snap = df[df["date"]==dt].dropna(subset=[beta_col])
    plt.figure(figsize=(7,4))
    plt.hist(snap[beta_col], bins=30)
    plt.title(f"Beta distribution — {pd.to_datetime(dt).date()}")
    plt.xlabel("Beta"); plt.ylabel("Count"); plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(outdir / "beta_hist.png"); plt.close()

def plot_beta_by_group(df: pd.DataFrame, dt, group_col: str, outname: str, outdir: Path):
    ensure_outdir(outdir)
    beta_col = _pick_beta_col(df)
    snap = df[df["date"]==dt].dropna(subset=[beta_col])
    g = snap.groupby(group_col, observed=True)[beta_col].mean().sort_values()
    plt.figure(figsize=(10,6))
    plt.barh(g.index.astype(str), g.values)
    plt.title(f"Average Beta by {group_col.capitalize()} — {pd.to_datetime(dt).date()}")
    plt.xlabel("Avg beta"); plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout(); plt.savefig(outdir / outname); plt.close()

def plot_scatter(df: pd.DataFrame, dt, xcol: str, ycol: str, outname: str, outdir: Path, s_col: Optional[str] = None):
    ensure_outdir(outdir)
    snap = df[df["date"]==dt].dropna(subset=[xcol,ycol]).copy()
    plt.figure(figsize=(6,5))
    if s_col and s_col in snap.columns:
        s = snap[s_col].fillna(0)
        med = s.median() if np.isfinite(s.median()) and s.median() > 0 else 1.0
        s = (s / med) * 10.0
        plt.scatter(snap[xcol], snap[ycol], s=s)
    else:
        plt.scatter(snap[xcol], snap[ycol])
    plt.xlabel(xcol); plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol} — {pd.to_datetime(dt).date()}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(outdir / outname); plt.close()

def plot_quintile_bar(df: pd.DataFrame, dt, factor: str, target: str, outname: str, outdir: Path):
    ensure_outdir(outdir)
    snap = df[df["date"]==dt].dropna(subset=[factor,target]).copy()
    q = pd.qcut(snap[factor].rank(method="first"), 5, labels=[1,2,3,4,5], duplicates="drop")
    res = snap.groupby(q, observed=True)[target].mean()
    plt.figure(figsize=(6,4))
    plt.bar(res.index.astype(str), res.values)
    plt.title(f"Avg {target} by {factor} quintile — {pd.to_datetime(dt).date()}")
    plt.xlabel(f"{factor} quintile (1=low,5=high)"); plt.ylabel(f"Avg {target}")
    plt.grid(True, axis="y", alpha=0.25)
    if "ret" in target:
        _fmt_percent(plt.gca())
    plt.tight_layout(); plt.savefig(outdir / outname); plt.close()

def plot_tstat_hist(df: pd.DataFrame, dt, outdir: Path):
    if "beta_tstat" not in df.columns:
        return
    ensure_outdir(outdir)
    snap = df[df["date"]==dt].dropna(subset=["beta_tstat"])
    if snap.empty:
        return
    plt.figure(figsize=(7,4))
    plt.hist(snap["beta_tstat"], bins=30)
    plt.title(f"Beta t-stat (rolling OLS) — {pd.to_datetime(dt).date()}")
    plt.xlabel("t-stat"); plt.ylabel("Count"); plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(outdir / "beta_tstat_hist.png"); plt.close()

def correlation_table(df: pd.DataFrame, dt, cols: list[str]) -> pd.DataFrame:
    snap = df[df["date"]==dt][cols].dropna()
    if snap.empty:
        return pd.DataFrame()
    pearson  = snap.corr(method="pearson")
    spearman = snap.corr(method="spearman")
    out = pd.concat({"pearson": pearson, "spearman": spearman}, axis=1)
    return out

# ==========================
# 4) Neutralization (optional)
# ==========================
def neutralize_factors_by(df: pd.DataFrame,
                          factor_cols: List[str],
                          by_cols: List[str],
                          date_col: str = "date") -> pd.DataFrame:
    """
    Cross-sectional OLS by date: regress each factor on one-hot(by_cols),
    keep residuals. Adds '<factor>_neu' columns.
    """
    _check_cols(df, [date_col] + factor_cols + by_cols, "neutralize_factors_by")
    out = df.copy()

    def _neutralize_day(day: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(day[by_cols], drop_first=False)
        if X.shape[1] == 0:
            return day
        X = X.astype(float)
        X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
        Xv = X.values
        for c in factor_cols:
            y = pd.to_numeric(day[c], errors="coerce").values
            mask = np.isfinite(y) & np.isfinite(Xv).all(axis=1)
            if mask.sum() < Xv.shape[1] + 1:
                day[f"{c}_neu"] = np.nan
                continue
            beta, *_ = np.linalg.lstsq(Xv[mask], y[mask], rcond=None)
            resid = np.full_like(y, np.nan, dtype=float)
            resid[mask] = y[mask] - Xv[mask] @ beta
            day[f"{c}_neu"] = resid
        return day

    out = out.groupby(date_col, group_keys=False).apply(_neutralize_day)
    return out

# ==========================
# 5) Orchestrator / Driver
# ==========================
def run_full_eda_from_exports(
    *,
    indir: Path = DEFAULT_INDIR,
    outdir: Path = DEFAULT_OUTDIR,
    winsorize: bool = True,
    zscore: bool = False,
    neutralize_by: Optional[List[str]] = None,  # e.g., ['country','supersector']
    compute_beta_tstat: bool = True,
    downcast: bool = True
) -> dict:
    """
    EDA pipeline that starts from DATA_ENG exports (no raw recomputation).
    Loads df_with_beta / factors_daily, market_series, factors_eom_z (if present).
    Writes charts, tables and an HTML index to outdir.
    Returns a dict with the main artifacts.
    """
    ensure_outdir(outdir)
    logging.info("Loading exports from %s …", Path(indir).resolve())
    loaded = _load_exports(indir)
    df = loaded["df_with_beta"].copy()
    mkt = loaded["market_series"]
    eom_z = loaded["factors_eom_z"]
    factors_list = loaded["factors_list"]
    last_dt = pd.to_datetime(loaded["last_date"])

    # Basic hygiene
    for c in ["date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.sort_values(["ticker","date"], kind="mergesort")
    df = _as_category(df, ["ticker","country","supersector"])

    # Choose ret_stock for visuals (prefer excess if present)
    ret_col = "excess_ret_1d_eur" if "excess_ret_1d_eur" in df.columns else "ret_1d_eur"
    df["ret_stock"] = pd.to_numeric(df.get("ret_stock", df[ret_col]), errors="coerce")

    # Optionally compute rolling beta t-stats (beta_252 must already exist)
    if compute_beta_tstat and "beta_252" in df.columns and "mkt_ret" in df.columns:
        logging.info("Computing rolling beta t-stats …")
        df = attach_beta_tstat(df, ret_col="ret_1d_eur" if "ret_1d_eur" in df.columns else ret_col,
                               mkt_col="mkt_ret", roll_win=252, min_periods=200,
                               out_col="beta_252", tstat_col="beta_tstat")

    # Factor selection
    if factors_list is None:
        # derive from columns if JSON list was not exported
        non_factors = {"date","ticker","country","supersector","price_eur","turnover_eur",
                       "ret_1d_eur","excess_ret_1d_eur","mkt_ret","ret_stock"}
        factors = sorted([c for c in df.columns if c not in non_factors and df[c].dtype.kind in "fc"])
    else:
        factors = [c for c in factors_list if c in df.columns]

    # Optional neutralization (adds *_neu)
    use_for_plots = factors.copy()
    if neutralize_by:
        keep_these = [c for c in neutralize_by if c in df.columns]
        if keep_these:
            logging.info("Neutralizing factors by %s …", keep_these)
            df = neutralize_factors_by(df, factors, keep_these)
            use_for_plots = [f"{c}_neu" for c in factors if f"{c}_neu" in df.columns]

    # Optional x-sec winsorize & zscore by date
    if winsorize and use_for_plots:
        logging.info("Winsorizing factors per date (1%%/99%%) …")
        df[use_for_plots] = (
            df.groupby("date", group_keys=False)[use_for_plots]
              .apply(lambda d: d.apply(winsorize_xsec))
        )
    if zscore and use_for_plots:
        logging.info("Z-scoring factors per date …")
        df[use_for_plots] = (
            df.groupby("date", group_keys=False)[use_for_plots]
              .apply(lambda d: d.apply(zscore_xsec))
        )

    if downcast:
        df = _downcast_float(df)

    # =================
    # 6) Visual outputs
    # =================
    logging.info("Charts …")
    plot_coverage(df, outdir)
    if mkt is not None:
        plot_market(mkt, outdir)
    else:
        plot_market(df, outdir)

    # Decide a robust last_dt (prefer provided)
    last_dt = pd.to_datetime(last_dt) if pd.notna(last_dt) else pd.to_datetime(df["date"]).max()

    plot_beta_hist(df, last_dt, outdir)
    plot_tstat_hist(df, last_dt, outdir)
    if "supersector" in df.columns:
        plot_beta_by_group(df, last_dt, "supersector", "beta_by_sector.png", outdir)
    if "country" in df.columns:
        plot_beta_by_group(df, last_dt, "country", "beta_by_country.png", outdir)

    # A few standard factor visuals if present
    def pick(base: str) -> str:
        return f"{base}_neu" if f"{base}_neu" in df.columns else base

    for base, fname in [
        ("vol_252d", "scatter_beta_vs_vol.png"),
        ("mom_12_1", "scatter_beta_vs_mom.png"),
        ("size_proxy", "scatter_beta_vs_size.png"),
    ]:
        xcol = pick(base)
        if xcol in df.columns and "beta_252" in df.columns:
            plot_scatter(df, last_dt, xcol, "beta_252", fname, outdir, s_col="adv_eur_20" if "adv_eur_20" in df.columns else None)

    for base, fname in [
        ("mom_12_1", "ret_by_momentum_quintile.png"),
        ("size_proxy", "ret_by_size_quintile.png"),
    ]:
        xcol = pick(base)
        if xcol in df.columns and "ret_stock" in df.columns:
            plot_quintile_bar(df, last_dt, xcol, "ret_stock", fname, outdir)

    # ==================
    # 7) Tables & CSVs
    # ==================
    logging.info("Tables & CSVs …")
    cols_for_corr = [c for c in ["beta_252", pick("mom_12_1"), pick("vol_252d"), pick("size_proxy"), "ret_stock"] if c in df.columns]
    corr = correlation_table(df, last_dt, cols_for_corr)
    if not corr.empty:
        corr.to_csv(outdir / "cross_section_correlations_latest.csv")

    # Snapshots
    snap_cols = [c for c in ["ticker","country","supersector","beta_252","mom_12_1","mom_12_1_neu",
                             "vol_252d","vol_252d_neu","size_proxy","size_proxy_neu","ret_stock"] if c in df.columns]
    snap = df.loc[df["date"]==last_dt, snap_cols].copy()
    if not snap.empty:
        if "beta_252" in snap.columns:
            snap.sort_values("beta_252", ascending=False).head(20).to_csv(outdir / "top20_beta.csv", index=False)
            snap.sort_values("beta_252", ascending=True ).head(20).to_csv(outdir / "bottom20_beta.csv", index=False)
        if "mom_12_1" in snap.columns:
            snap.sort_values("mom_12_1", ascending=False).head(20).to_csv(outdir / "top20_momentum.csv", index=False)
            snap.sort_values("mom_12_1", ascending=True ).head(20).to_csv(outdir / "bottom20_momentum.csv", index=False)
        if "size_proxy" in snap.columns:
            snap.sort_values("size_proxy", ascending=False).head(20).to_csv(outdir / "top20_size.csv", index=False)
            snap.sort_values("size_proxy", ascending=True ).head(20).to_csv(outdir / "bottom20_size.csv", index=False)
        if "beta_tstat" in df.columns:
            df.loc[df["date"]==last_dt, ["ticker","beta_tstat"]].sort_values("beta_tstat", ascending=False).head(50) \
              .to_csv(outdir / "top50_beta_tstat.csv", index=False)

    cov_by_date = df.groupby("date")["ticker"].nunique().rename("n_tickers").reset_index()
    cov_by_date.to_csv(outdir / "coverage_by_date.csv", index=False)

    nan_cols = ["mom_12_1","mom_12_1_neu","vol_252d","vol_252d_neu","size_proxy","size_proxy_neu","beta_252","ret_stock"]
    nan_share = summarize_nan_share_by_date(df, nan_cols)
    if not nan_share.empty:
        nan_share.to_csv(outdir / "nan_share_by_feature.csv")

    # Parquet export (fast reload)
    df.to_parquet(outdir / "panel_with_beta.parquet", index=False)

    # Small HTML report linking all outputs
    _write_html_index(outdir, last_dt)

    logging.info("Done. Saved outputs to %s", outdir.resolve())
    return {
        "model_df_v1": loaded.get("model_df_v1"),
        "model_df_v2": loaded.get("model_df_v2"),
        "market_series": mkt,
        "df_with_beta": df,
        "corr_latest": corr,
        "last_date": last_dt,
    }

def _write_html_index(outdir: Path, last_dt: pd.Timestamp) -> None:
    imgs = [
        "coverage_count.png",
        "market_index.png",
        "beta_hist.png",
        "beta_tstat_hist.png",
        "beta_by_sector.png",
        "beta_by_country.png",
        "scatter_beta_vs_vol.png",
        "scatter_beta_vs_mom.png",
        "scatter_beta_vs_size.png",
        "ret_by_momentum_quintile.png",
        "ret_by_size_quintile.png",
    ]
    files = [
        "market_series.csv",
        "cross_section_correlations_latest.csv",
        "top20_beta.csv",
        "bottom20_beta.csv",
        "top20_momentum.csv",
        "bottom20_momentum.csv",
        "top20_size.csv",
        "bottom20_size.csv",
        "top50_beta_tstat.csv",
        "coverage_by_date.csv",
        "nan_share_by_feature.csv",
        "panel_with_beta.parquet",
    ]
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>EDA Outputs</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;} img{border:1px solid #ddd;}</style>",
        "</head><body>",
        f"<h1>EDA Outputs — {pd.to_datetime(last_dt).date()}</h1>",
        "<h2>Charts</h2>",
    ]
    for img in imgs:
        p = outdir / img
        if p.exists():
            html += [f"<h3>{img}</h3>", f"<img src='{img}' style='max-width:100%;height:auto;'/>"]
    html += ["<h2>Tables & Files</h2>", "<ul>"]
    for c in files:
        p = outdir / c
        if p.exists():
            html += [f"<li><a href='{c}'>{c}</a></li>"]
    html += ["</ul>", "</body></html>"]
    (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")

# ==========================
# 6) CLI
# ==========================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA pipeline using DATA_ENG exports")
    p.add_argument("--indir", type=Path, default=DEFAULT_INDIR, help="Input directory with DATA_ENG exports")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for EDA artifacts")
    p.add_argument("--no-winsorize", action="store_true", help="Disable cross-sectional winsorization")
    p.add_argument("--zscore", action="store_true", help="Enable cross-sectional z-scoring")
    p.add_argument("--neutralize", nargs="*", default=None, help="Neutralize factors by these columns (e.g., country supersector)")
    p.add_argument("--no-beta-tstat", action="store_true", help="Skip beta t-stat computation")
    p.add_argument("--no-downcast", action="store_true", help="Skip float downcasting to save memory")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_full_eda_from_exports(
        indir=args.indir,
        outdir=args.outdir,
        winsorize=not args.no_winsorize,
        zscore=args.zscore,
        neutralize_by=args.neutralize,
        compute_beta_tstat=not args.no_beta_tstat,
        downcast=not args.no_downcast,
    )
