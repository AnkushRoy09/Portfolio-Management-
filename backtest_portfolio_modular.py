#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
backtest_portfolio_modular.py
-----------------------------
Standalone monthly backtester that uses your shared weighting module:
    portfolio_construction.build_target_weights(...)

Inputs (expected from your pipeline):
- C:/Users/USER/Portfolio Management/final_sector_valuation.csv
- C:/Users/USER/Portfolio Management/time_series/prices_adjusted.csv
- C:/Users/USER/Portfolio Management/time_series/benchmark_SPY.csv
- C:/Users/USER/Portfolio Management/time_series/risk_free_DTB3.csv  (optional)
- C:/Users/USER/Portfolio Management/time_series/monthly_signals_lagged.csv

Outputs:
- time_series/equity_curve.csv
- time_series/equity_curve_vs_benchmark.png
- time_series/performance_summary.csv
- outputs/weights_history.csv
- outputs/portfolio_holdings_latest.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the shared weighting logic (ensure portfolio_construction.py is on the same path)
try:
    from portfolio_construction import build_target_weights
except ImportError as e:
    raise SystemExit("Could not import 'portfolio_construction'. "
                     "Place portfolio_construction.py next to this script "
                     "or add it to PYTHONPATH.") from e


# ------------------------- Loaders & helpers -------------------------

def normalize_ticker(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().upper()
    return t.replace(".", "-")


def load_prices(prices_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(prices_csv, low_memory=False)
    # Use first column as date if not labeled
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.set_index(date_col)
    # Parse dates safely
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    # Normalize columns to tickers and coerce numeric
    df.columns = [normalize_ticker(c) for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", axis=1)
    return df


def load_benchmark(bench_csv: Path) -> pd.Series:
    df = pd.read_csv(bench_csv, low_memory=False)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    # pick Adjusted Close if available, else the first numeric column
    candidates = [c for c in df.columns if c.lower() in ("adj close", "adj_close", "adjusted close", "close")]
    col = candidates[0] if candidates else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.name = "Benchmark"
    return s


def load_rf(rf_csv: Path) -> Optional[pd.Series]:
    if not rf_csv.exists():
        return None
    df = pd.read_csv(rf_csv, low_memory=False)
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    col = "RF" if "RF" in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.name = "RF"
    return s


def load_sector_cap_map(final_csv: Path) -> (Dict[str, str], Dict[str, float]):
    sector_map, cap_map = {}, {}
    if not final_csv.exists():
        return sector_map, cap_map
    df = pd.read_csv(final_csv, low_memory=False)
    needed = {"Ticker", "Sector", "Market Cap"}
    if not needed.issubset(df.columns):
        return sector_map, cap_map
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    sector_map = dict(zip(df["Ticker"], df["Sector"]))
    cap_map = dict(zip(df["Ticker"], pd.to_numeric(df["Market Cap"], errors="coerce").fillna(0.0)))
    return sector_map, cap_map


def load_signals(sig_csv: Path) -> pd.DataFrame:
    sig = pd.read_csv(sig_csv, index_col=0, parse_dates=True, low_memory=False)
    sig.index = pd.to_datetime(sig.index, errors="coerce")
    sig = sig[~sig.index.isna()].sort_index()
    sig.columns = [normalize_ticker(c) for c in sig.columns]
    for c in sig.columns:
        sig[c] = pd.to_numeric(sig[c], errors="coerce")
    return sig.fillna(0).astype("int16")


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    px_m = prices.resample("ME").last()
    return px_m.pct_change().fillna(0.0)


def performance_summary(eq: pd.Series, bench_eq: pd.Series, rf_daily: Optional[pd.Series] = None) -> pd.DataFrame:
    idx = eq.index.intersection(bench_eq.index)
    eq = eq.reindex(idx)
    bench_eq = bench_eq.reindex(idx)

    r = eq.pct_change().dropna()
    rb = bench_eq.pct_change().dropna()

    af = 12.0  # monthly
    cagr = (eq.iloc[-1] / eq.iloc[0])**(af / len(r)) - 1 if len(r) > 0 else np.nan
    vol = r.std() * np.sqrt(af) if len(r) > 1 else np.nan

    if rf_daily is not None and not rf_daily.empty:
        rf_m = rf_daily.resample("ME").mean().reindex(r.index).fillna(method="ffill") / 12.0
        ex = r.sub(rf_m, fill_value=0.0)
        sharpe = (ex.mean() / ex.std() * np.sqrt(af)) if ex.std() > 0 else np.nan
    else:
        sharpe = (r.mean() / r.std() * np.sqrt(af)) if r.std() > 0 else np.nan

    roll_max = eq.cummax()
    mdd = (eq / roll_max - 1.0).min() if not eq.empty else np.nan

    rr = r.reindex(rb.index).dropna()
    rbb = rb.reindex(rr.index).dropna()
    te = (rr - rbb).std() * np.sqrt(af) if len(rr) > 1 else np.nan

    return pd.DataFrame([{
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Tracking Error": te
    }])


def plot_equity(eq: pd.Series, bench: pd.Series, out_png: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 6))
    eq.plot(label="Strategy")
    bench.plot(label="Benchmark")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.title("Equity Curve: Strategy vs. Benchmark")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ------------------------------ Main -------------------------------

def main():
    p = argparse.ArgumentParser()
    # Use forward slashes by default to avoid \U issues on Windows
    p.add_argument("--base-dir", type=str, default="C:/Users/USER/Portfolio Management",
                   help="Base directory where pipeline outputs live")
    p.add_argument("--signals-file", type=str, default="monthly_signals_lagged.csv",
                   help="Signals CSV filename under time_series/")
    p.add_argument("--benchmark", type=str, default="SPY",
                   help="Benchmark suffix used in file name under time_series (e.g., benchmark_SPY.csv)")
    p.add_argument("--weighting", choices=["vol", "cap", "sharpe"], default="vol",
                   help="Weighting scheme (no equal-weight allowed)")
    p.add_argument("--market-neutral", action="store_true",
                   help="Use 50/50 long/short (net ~0) if shorts exist")
    p.add_argument("--tc-bps", type=float, default=10.0,
                   help="Transaction cost bps per unit turnover per rebalance")
    p.add_argument("--max-weight", type=float, default=0.05, help="Max single-name weight")
    p.add_argument("--max-sector-weight", type=float, default=0.25, help="Max sector gross weight")
    p.add_argument("--lookback-months", type=int, default=3,
                   help="Lookback months for vol/Sharpe calculations")
    p.add_argument("--start", type=str, default=None, help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--rf-annual", type=float, default=None,
                   help="Annual risk-free (e.g., 0.05 for 5%) used only in Sharpe weighting")
    p.add_argument("--carry-forward-empty", action="store_true",
                   help="If a month's weights are empty, carry forward previous month's weights")
    args = p.parse_args()

    base = Path(args.base_dir)
    ts_dir = base / "time_series"
    out_hold = base / "outputs"
    ts_dir.mkdir(parents=True, exist_ok=True)
    out_hold.mkdir(parents=True, exist_ok=True)

    prices_csv = ts_dir / "prices_adjusted.csv"
    sig_csv = ts_dir / args.signals_file
    bench_csv = ts_dir / f"benchmark_{args.benchmark}.csv"
    rf_csv = ts_dir / "risk_free_DTB3.csv"
    final_csv = base / "final_sector_valuation.csv"

    missing = [str(x) for x in [prices_csv, sig_csv, bench_csv] if not x.exists()]

    if missing:
        print("Error: Missing required files:\n - " + "\n - ".join(missing))
        return

    print("Reading prices:", prices_csv)
    prices = load_prices(prices_csv)

    if args.start:
        prices = prices[prices.index >= pd.to_datetime(args.start)]
    if args.end:
        prices = prices[prices.index <= pd.to_datetime(args.end)]

    print("Reading signals:", sig_csv)
    signals = load_signals(sig_csv)

    print("Reading benchmark:", bench_csv)
    bench = load_benchmark(bench_csv)

    print("Loading sector & cap maps:", final_csv)
    sector_map, cap_map = load_sector_cap_map(final_csv)

    print("Optional RF:", rf_csv)
    rf_daily = load_rf(rf_csv)

    # Keep only tickers present in both prices and signals
    common = sorted(set(prices.columns).intersection(set(signals.columns)))
    if not common:
        print("Error: No overlapping tickers between prices and signals.")
        print("Example prices:", list(prices.columns)[:10])
        print("Example signals:", list(signals.columns)[:10])
        return
    prices = prices[common]
    signals = signals[common]

    # Compute monthly returns and align dates
    px_m = prices.resample("ME").last()
    rets = px_m.pct_change().fillna(0.0)
    bench_m = bench.resample("ME").last().pct_change().fillna(0.0)

    common_idx = rets.index.intersection(signals.index).intersection(bench_m.index)
    rets = rets.reindex(common_idx)
    signals = signals.reindex(common_idx).fillna(0).astype(int)
    bench_m = bench_m.reindex(common_idx)

    # Backtest loop
    equity = 1.0
    eq_points = []
    w_prev: Dict[str, float] = {}
    weights_history = []

    dates = list(common_idx)
    for i in range(len(dates) - 1):
        t0, t1 = dates[i], dates[i + 1]
        sig_row = signals.loc[t0]

        # lookback window
        rets_win = None
        if args.lookback_months > 0 and i >= args.lookback_months:
            prev_span = dates[i - args.lookback_months:i]
            rets_win = rets.reindex(prev_span)

        # build weights via shared module
        w_t = build_target_weights(
            sig_row=sig_row,
            prices_window=rets_win,
            sector_map=sector_map,
            cap_map=cap_map,
            weighting=args.weighting,
            market_neutral=args.market_neutral,
            max_w=args.max_weight,
            max_sector_w=args.max_sector_weight,
            rf_annual=args.rf_annual
        )

        # Carry-forward safeguard if empty
        carried = False
        if (not w_t) and args.carry_forward_empty and w_prev:
            w_t = dict(w_prev)
            carried = True

        # transaction costs from turnover
        turnover = sum(abs(w_t.get(k, 0.0) - w_prev.get(k, 0.0)) for k in set(w_t) | set(w_prev))
        tc = args.tc_bps / 1e4 * turnover

        # realize return for period t0->t1
        r1 = rets.loc[t1].fillna(0.0)
        port_ret = float(pd.Series(w_t).reindex(rets.columns).fillna(0.0).dot(r1))
        equity *= max((1.0 + port_ret) * (1.0 - tc), 0.0)

        eq_points.append((t1, equity))
        weights_history.append({"Date": t0, **w_t})
        w_prev = w_t

    eq = pd.Series([v for _, v in eq_points], index=[d for d, _ in eq_points], name="Strategy")
    bench_eq = (1.0 + bench_m).cumprod().reindex(eq.index)

    # Save outputs
    ts_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame({"Strategy": eq, "Benchmark": bench_eq})
    eq_csv = ts_dir / "equity_curve.csv"
    eq_df.to_csv(eq_csv)
    print("Saved:", eq_csv)

    perf = performance_summary(eq, bench_eq, rf_daily=rf_daily)
    perf_csv = ts_dir / "performance_summary.csv"
    perf.to_csv(perf_csv, index=False)
    print("Saved:", perf_csv)

    png = ts_dir / "equity_curve_vs_benchmark.png"
    plot_equity(eq, bench_eq, png)
    print("Saved:", png)

    if weights_history:
        wdf = pd.DataFrame(weights_history).set_index("Date")
        out_w = out_hold / "weights_history.csv"
        wdf.to_csv(out_w)
        last = wdf.iloc[[-1]].T
        last.columns = ["Weight"]
        out_last = out_hold / "portfolio_holdings_latest.csv"
        last.to_csv(out_last)
        print("Saved:", out_w)
        print("Saved:", out_last)

    print("\nBacktest complete.")


if __name__ == "__main__":
    import sys

    # --- Default configuration (no need to type manually) ---
    default_args = [
        "--base-dir", "C:/Users/USER/Portfolio Management",
        "--weighting", "sharpe",
        "--rf-annual", "0.05",
        "--carry-forward-empty"
    ]

    # Only add defaults if the user didn't pass custom args
    if len(sys.argv) == 1:
        sys.argv.extend(default_args)

    main()
