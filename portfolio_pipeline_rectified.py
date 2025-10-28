#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Management Pipeline (Rectified)
----------------------------------------
Steps:
1) Scrape S&P 500 tickers from Wikipedia
2) Select up to N companies per sector (default 100)
3) Fetch metrics from Yahoo Finance using robust fallbacks
4) Compute sector medians
5) Classify each company vs. sector medians (±20% rule)
6) Create LLH/HHL combo valuation status
7) Export CSVs, an Excel workbook, and a PNG chart

Requirements:
  pandas, yfinance, requests, matplotlib, openpyxl

Usage:
  python portfolio_pipeline_rectified.py --per-sector 100 --threshold 0.2 --outdir out
"""

import argparse
import math
import random
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (compatible; PortfolioPipeline/1.1; +https://example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

def fetch_sp500_table(url=WIKI_URL, session: requests.Session = SESSION, timeout=30) -> pd.DataFrame:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["Ticker", "Company", "Sector"]
    # yfinance expects BRK.B style as BRK-B, BF.B as BF-B, etc.
    df["Ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
    return df

def sample_per_sector(sp500: pd.DataFrame, n=100, random_state=42) -> pd.DataFrame:
    return (
        sp500.groupby("Sector", group_keys=False)
        .apply(lambda g: g.sample(min(n, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )

def _safe_get(d, *keys):
    for k in keys:
        if d is None:
            return None
        if isinstance(d, dict) and k in d and pd.notna(d.get(k)):
            return d.get(k)
    return None

def get_metrics_for_ticker(ticker: str, retries=3, sleep_base=0.6):
    """Return dict with Market Cap, P/E, P/B, ROE using robust fallbacks."""
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker, session=SESSION)

            # Prefer new API surface
            info = {}
            try:
                info = tk.get_info()
            except Exception:
                info = {}

            try:
                fast = tk.fast_info
            except Exception:
                fast = {}

            last_price = _safe_get(fast, "last_price") or _safe_get(info, "currentPrice", "regularMarketPrice")

            # Market Cap
            mcap = _safe_get(info, "marketCap") or _safe_get(fast, "market_cap")

            # P/E
            pe = (
                _safe_get(info, "trailingPE", "peRatio", "trailingPe")
                or _safe_get(fast, "trailing_pe", "pe")
                or ((last_price / _safe_get(info, "trailingEps"))
                    if last_price and _safe_get(info, "trailingEps") not in (None, 0) else None)
            )

            # P/B
            pb = (
                _safe_get(info, "priceToBook")
                or _safe_get(fast, "price_to_book")
                or ((last_price / _safe_get(info, "bookValue"))
                    if last_price and _safe_get(info, "bookValue") not in (None, 0) else None)
            )

            # ROE
            roe = _safe_get(info, "returnOnEquity", "roe") or _safe_get(fast, "return_on_equity")

            # Fallback from financials: ROE = Net Income / Total Equity
            if roe is None:
                try:
                    fin = tk.get_financials(freq="yearly")
                    bs = tk.get_balance_sheet(freq="yearly")
                    if isinstance(fin, pd.DataFrame) and isinstance(bs, pd.DataFrame) and not fin.empty and not bs.empty:
                        fin_idx = {str(i).lower(): i for i in fin.index}
                        bs_idx = {str(i).lower(): i for i in bs.index}
                        ni_key = next((fin_idx[k] for k in fin_idx if "net income" in k), None)
                        eq_key = next((bs_idx[k] for k in bs_idx if ("total stockholder" in k and "equity" in k) or "total equity" in k), None)
                        if ni_key and eq_key:
                            ni = pd.to_numeric(fin.loc[ni_key].dropna().iloc[0], errors="coerce")
                            eq = pd.to_numeric(bs.loc[eq_key].dropna().iloc[0], errors="coerce")
                            if pd.notna(ni) and pd.notna(eq) and eq not in (0,):
                                roe = float(ni) / float(eq)
                except Exception:
                    pass

            def _num(x):
                try:
                    return float(x)
                except Exception:
                    return None

            mcap = _num(mcap)
            pe = _num(pe)
            pb = _num(pb)
            roe = _num(roe)
            if roe is not None:
                roe = max(-1.0, min(1.0, roe))

            metrics = {"Market Cap": mcap, "P/E Ratio": pe, "P/B Ratio": pb, "ROE": roe}
            if not any(v is not None for v in metrics.values()):
                raise RuntimeError("Empty metrics")
            return metrics

        except Exception:
            time.sleep(sleep_base * (2 ** attempt) + random.random() * 0.3)
    return {}

def fetch_all_metrics(selected: pd.DataFrame, pause=0.4):
    rows = []
    for i, row in selected.reset_index(drop=True).iterrows():
        tic = row["Ticker"]
        met = get_metrics_for_ticker(tic)
        if met and any(pd.notna(list(met.values()))):
            met.update({"Company": row["Company"], "Ticker": tic, "Sector": row["Sector"]})
            rows.append(met)
            print(f"[{i+1}/{len(selected)}] {tic:>6} ✓")
        else:
            print(f"[{i+1}/{len(selected)}] {tic:>6} skipped (no data)")
        time.sleep(pause)
    return pd.DataFrame(rows)

def compute_sector_medians(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Sector")[["P/E Ratio", "P/B Ratio", "ROE"]].median(numeric_only=True).reset_index()

def classify_metric(value, median, threshold=0.2):
    if pd.isna(value) or pd.isna(median):
        return "N/A"
    try:
        v = float(value)
        m = float(median)
    except Exception:
        return "N/A"
    if m == 0 or math.isclose(m, 0.0):
        return "N/A"
    if v > m * (1 + threshold):
        return "High"
    elif v < m * (1 - threshold):
        return "Low"
    else:
        return "Neutral"

def add_metric_signals(df: pd.DataFrame, med: pd.DataFrame, threshold=0.2) -> pd.DataFrame:
    df = pd.merge(df, med, on="Sector", suffixes=("", "_Median"))
    df["P/E Signal"] = df.apply(lambda r: classify_metric(r["P/E Ratio"], r["P/E Ratio_Median"], threshold), axis=1)
    df["P/B Signal"] = df.apply(lambda r: classify_metric(r["P/B Ratio"], r["P/B Ratio_Median"], threshold), axis=1)
    df["ROE Signal"] = df.apply(lambda r: classify_metric(r["ROE"], r["ROE_Median"], threshold), axis=1)
    return df

def add_combo_status(df: pd.DataFrame) -> pd.DataFrame:
    df["Signal_Combo"] = (
        df["P/E Signal"].astype(str).str[0]
        + df["P/B Signal"].astype(str).str[0]
        + df["ROE Signal"].astype(str).str[0]
    )
    df["Valuation Status"] = df["Signal_Combo"].map({"LLH": "Undervalued", "HHL": "Overvalued"}).fillna("Neutral")
    return df

def save_all_outputs(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Sort and save
    df.sort_values(by=["Sector", "Market Cap"], ascending=[True, False]).to_csv(outdir / "portfolio.csv", index=False)

    # Sector medians
    sect_med = df.groupby("Sector")[["P/E Ratio", "P/B Ratio", "ROE"]].median(numeric_only=True).reset_index()
    sect_med.to_csv(outdir / "sector_medians.csv", index=False)

    # Largest by sector (already captured in sorted portfolio; also save explicit file)
    df.sort_values(by=["Sector", "Market Cap"], ascending=[True, False]).to_csv(outdir / "largest_company_per_sector.csv", index=False)

    # Final with signals
    df.to_csv(outdir / "final_sector_valuation.csv", index=False)

    # Excel workbook
    undervalued = df[df["Valuation Status"] == "Undervalued"]
    overvalued = df[df["Valuation Status"] == "Overvalued"]
    summary_tbl = df.groupby(["Sector", "Valuation Status"]).size().unstack(fill_value=0)

    with pd.ExcelWriter(outdir / "valuation_summary.xlsx", engine="openpyxl") as w:
        undervalued.to_excel(w, sheet_name="Undervalued", index=False)
        overvalued.to_excel(w, sheet_name="Overvalued", index=False)
        summary_tbl.to_excel(w, sheet_name="Summary")

    # Chart: one figure, no custom colors
    plt.figure(figsize=(12, 6))
    summary_tbl.plot(kind="bar", stacked=True)
    plt.xlabel("Sector")
    plt.ylabel("Number of Companies")
    plt.title("Valuation Status Distribution by Sector")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "valuation_status_by_sector.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sector", type=int, default=100, help="Max companies per sector")
    parser.add_argument("--threshold", type=float, default=0.2, help="± threshold for High/Low classification")
    
    # parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    args = parser.parse_args()

    # Force-save all outputs to your Portfolio Management folder
    outdir = Path(r"C:\Users\USER\Portfolio Management")
    outdir.mkdir(parents=True, exist_ok=True)
    print("Fetching S&P 500 table from Wikipedia...")
    sp500 = fetch_sp500_table()
    print(f"Retrieved {len(sp500)} rows across {sp500['Sector'].nunique()} sectors.")

    print(f"Sampling up to {args.per_sector} per sector...")
    selected = sample_per_sector(sp500, n=args.per_sector)
    print(f"Selected {len(selected)} tickers across {selected['Sector'].nunique()} sectors.")

    print("Fetching metrics from Yahoo Finance (robust mode)...")
    metrics = fetch_all_metrics(selected)

    if metrics.empty:
        print("No metrics retrieved. Exiting without outputs.")
        return

    # Compute medians and signals
    sector_medians = compute_sector_medians(metrics)
    merged = add_metric_signals(metrics, sector_medians, threshold=args.threshold)
    merged = add_combo_status(merged)

    # Save everything
    save_all_outputs(merged, outdir)

    print("\n Done. Files written to:", outdir.resolve())
    for f in [
        "largest_company_per_sector.csv",
        "portfolio.csv",
        "sector_medians.csv",
        "final_sector_valuation.csv",
        "valuation_summary.xlsx",
        "valuation_status_by_sector.png",
    ]:
        print(" -", f)

if __name__ == "__main__":

    
    main()
