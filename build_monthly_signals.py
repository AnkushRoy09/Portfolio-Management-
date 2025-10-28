#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_monthly_signals_active.py
--------------------------------
Value signals with *minimum activity guarantees*.

Primary logic:
  1) Compute a discrete value score from P/E, P/B, ROE signals:
       Low  -> +1,  High -> -1, N/A/Neutral -> 0
       score = pe_score + pb_score + roe_score in [-3, 3]
  2) Default selection (2-of-3 rule):
       Long  (+1) if score >=  2
       Short (-1) if score <= -2
  3) If active names are below minimums, progressively relax:
       - First relax to score >= 1 (longs) and <= -1 (shorts)
       - If still short, pick top/bottom by score until min targets are met

Inputs (override with --base-dir):
  C:/Users/USER/Portfolio Management/final_sector_valuation.csv
  C:/Users/USER/Portfolio Management/time_series/prices_adjusted.csv

Outputs:
  time_series/monthly_signals.csv
  time_series/monthly_signals_lagged.csv
  time_series/signals_diagnostics_active.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------- Helpers ---------------------

def normalize_ticker(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().upper()
    return t.replace(".", "-")


def load_prices(prices_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(prices_csv, low_memory=False)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df.columns = [normalize_ticker(c) for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all", axis=1)


def load_final(final_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(final_csv, low_memory=False)
    if "Ticker" not in df.columns:
        raise ValueError("Ticker column is required in final_sector_valuation.csv")
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    # keep needed columns if present
    keep = ["Ticker", "Sector", "P/E Signal", "P/B Signal", "ROE Signal", "Valuation Status"]
    cols = [c for c in keep if c in df.columns]
    return df[cols].drop_duplicates(subset=["Ticker"], keep="first")


def make_month_end_index(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return prices.resample("ME").last().index


def sig_to_score(x: str, metric: str) -> int:
    """Map textual signals to numeric score contributions."""
    if not isinstance(x, str):
        return 0
    s = x.strip().lower()
    if metric in ("pe", "pb"):
        if s == "low":  return +1
        if s == "high": return -1
    if metric == "roe":
        if s == "high": return +1
        if s == "low":  return -1
    return 0


def compute_value_scores(df_final: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns: Ticker, score in [-3,3]."""
    tmp = df_final.set_index("Ticker")
    pe = tmp.get("P/E Signal")
    pb = tmp.get("P/B Signal")
    roe = tmp.get("ROE Signal")
    pe_s = pe.map(lambda x: sig_to_score(x, "pe")) if pe is not None else 0
    pb_s = pb.map(lambda x: sig_to_score(x, "pb")) if pb is not None else 0
    roe_s = roe.map(lambda x: sig_to_score(x, "roe")) if roe is not None else 0
    score = pd.Series(pe_s, dtype="int16").fillna(0) + pd.Series(pb_s, dtype="int16").fillna(0) + pd.Series(roe_s, dtype="int16").fillna(0)
    out = pd.DataFrame({"Ticker": score.index, "score": score.values})
    return out


def build_active_signal(score_df: pd.DataFrame, prices_cols, min_longs=30, min_shorts=30) -> pd.Series:
    """Build cross-sectional {-1,0,1} signals ensuring minimum activity and restricting to tickers present in prices."""
    # universe intersection with prices
    score_df = score_df[score_df["Ticker"].isin(prices_cols)].copy()
    score_df = score_df.dropna(subset=["score"])
    score_df["score"] = score_df["score"].astype(int)

    # 2-of-3 default
    longs = score_df[score_df["score"] >= 2]["Ticker"].tolist()
    shorts = score_df[score_df["score"] <= -2]["Ticker"].tolist()

    # Relax if below minimums
    if len(longs) < min_longs or len(shorts) < min_shorts:
        longs = score_df[score_df["score"] >= 1]["Ticker"].tolist()
        shorts = score_df[score_df["score"] <= -1]["Ticker"].tolist()

    # If still short, pick by top/bottom scores
    if len(longs) < min_longs:
        extra = score_df.sort_values("score", ascending=False)["Ticker"].tolist()
        longs = list(dict.fromkeys(longs + extra))[:min_longs]
    if len(shorts) < min_shorts:
        extra = score_df.sort_values("score", ascending=True)["Ticker"].tolist()
        shorts = list(dict.fromkeys(shorts + extra))[:min_shorts]

    # Construct signal vector
    all_tickers = sorted(set(score_df["Ticker"].tolist()))
    sig = pd.Series(0, index=all_tickers, dtype="int8")
    sig.loc[longs] = 1
    sig.loc[shorts] = -1
    return sig


def expand_monthly(sig_xsec: pd.Series, month_idx: pd.DatetimeIndex) -> pd.DataFrame:
    row = sig_xsec.reindex(sig_xsec.index).fillna(0).astype("int8")
    m = pd.DataFrame(np.tile(row.values, (len(month_idx), 1)),
                     index=month_idx, columns=row.index).astype("int8")
    return m


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default="C:/Users/USER/Portfolio Management", help="Base dir for inputs/outputs")
    ap.add_argument("--min-longs", type=int, default=30, help="Minimum number of long candidates")
    ap.add_argument("--min-shorts", type=int, default=30, help="Minimum number of short candidates")
    args = ap.parse_args()

    base = Path(args.base_dir)
    final_csv = base / "final_sector_valuation.csv"
    prices_csv = base / "time_series" / "prices_adjusted.csv"

    print("Reading:", final_csv)
    df_final = load_final(final_csv)

    print("Reading:", prices_csv)
    prices = load_prices(prices_csv)

    # Month index
    month_idx = make_month_end_index(prices)

    # Value scores and active cross-section
    scores = compute_value_scores(df_final)
    sig_xsec = build_active_signal(scores, prices.columns, min_longs=args.min_longs, min_shorts=args.min_shorts)

    # Monthly matrix and lag
    signals = expand_monthly(sig_xsec, month_idx)
    signals_lag = signals.shift(1).fillna(0).astype("int8")

    # Outputs
    ts_dir = base / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    out_a = ts_dir / "monthly_signals.csv"
    out_b = ts_dir / "monthly_signals_lagged.csv"
    signals.to_csv(out_a)
    signals_lag.to_csv(out_b)

    # Diagnostics
    diag = pd.DataFrame({
        "ActiveLongs": (signals == 1).sum(axis=1),
        "ActiveShorts": (signals == -1).sum(axis=1),
        "TotalActive": ((signals == 1).sum(axis=1) + (signals == -1).sum(axis=1))
    })
    diag_path = ts_dir / "signals_diagnostics_active.csv"
    diag.to_csv(diag_path)

    print("\n=== Diagnostics ===")
    print("Tickers in prices universe:", len(prices.columns))
    print("Cross-sectional tickers   :", signals.shape[1])
    print("First 6 months of active counts:")
    print(diag.head(6))
    print("\nSaved:")
    print(" -", out_a)
    print(" -", out_b)
    print(" -", diag_path)



def main():
    # defaults you won't need to type anymore
    base_dir = Path("C:/Users/USER/Portfolio Management")
    min_longs = 40
    min_shorts = 40

    final_csv = base_dir / "final_sector_valuation.csv"
    prices_csv = base_dir / "time_series" / "prices_adjusted.csv"

    print("Reading:", final_csv)
    df_final = load_final(final_csv)

    print("Reading:", prices_csv)
    prices = load_prices(prices_csv)

    # Month index
    month_idx = make_month_end_index(prices)

    # Value scores and active cross-section
    scores = compute_value_scores(df_final)
    sig_xsec = build_active_signal(scores, prices.columns, min_longs=min_longs, min_shorts=min_shorts)

    # Monthly matrix and lag
    signals = expand_monthly(sig_xsec, month_idx)
    signals_lag = signals.shift(1).fillna(0).astype("int8")

    # Outputs
    ts_dir = base_dir / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    out_a = ts_dir / "monthly_signals.csv"
    out_b = ts_dir / "monthly_signals_lagged.csv"
    signals.to_csv(out_a)
    signals_lag.to_csv(out_b)

    # Diagnostics
    diag = pd.DataFrame({
        "ActiveLongs": (signals == 1).sum(axis=1),
        "ActiveShorts": (signals == -1).sum(axis=1),
        "TotalActive": ((signals == 1).sum(axis=1) + (signals == -1).sum(axis=1))
    })
    diag_path = ts_dir / "signals_diagnostics_active.csv"
    diag.to_csv(diag_path)

    print("\n=== Diagnostics ===")
    print("Tickers in prices universe:", len(prices.columns))
    print("Cross-sectional tickers   :", signals.shape[1])
    print("First 6 months of active counts:")
    print(diag.head(6))
    print("\nSaved:")
    print(" -", out_a)
    print(" -", out_b)
    print(" -", diag_path)


if __name__ == "__main__":
    main()
