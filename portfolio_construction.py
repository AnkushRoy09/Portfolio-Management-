#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_construction.py
-------------------------
Shared portfolio weighting logic for your Value strategy.

Provides:
    build_target_weights(sig_row, prices_window, sector_map, cap_map=None,
                         weighting="vol", market_neutral=False,
                         max_w=0.05, max_sector_w=0.25, rf_annual=None) -> dict

Weighting options:
    - "vol"    : inverse-volatility weighting
    - "cap"    : market-cap proportional weighting (uses cap_map)
    - "sharpe" : Sharpe-based weighting with multiplier ladder

This module is imported by backtesting and execution scripts so you keep one
source of truth for position sizing.
"""

from typing import Dict, Optional, Iterable
import numpy as np
import pandas as pd


# ---------------------------- Utilities ----------------------------

def normalize_ticker(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().upper()
    return t.replace(".", "-")


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    """Best-effort guess of sampling frequency to convert annual RF to per-period.
    If monthly-like, return ~12. If daily-like, return ~252. Else derive from median spacing.
    """
    if index is None or len(index) < 3:
        return 12.0
    diffs = np.diff(index.values).astype('timedelta64[D]').astype(float)
    med_days = float(np.nanmedian(diffs)) if len(diffs) else 30.0
    if med_days <= 2.5:   # daily
        return 252.0
    if 25.0 <= med_days <= 35.0:  # monthly-ish
        return 12.0
    if 80.0 <= med_days <= 110.0:  # quarterly-ish
        return 4.0
    # Fallback
    return max(1.0, round(365.0 / max(1.0, med_days)))


def _to_numpy_series(values: Iterable[float], names: Iterable[str]) -> pd.Series:
    s = pd.Series(values, index=list(names), dtype="float64")
    s.replace([np.inf, -np.inf], np.nan, inplace=True)
    return s.fillna(0.0)


# ----------------------- Core weighting blocks ----------------------

def _inverse_vol_weights(names, rets_window: pd.DataFrame, eps: float = 1e-8) -> Dict[str, float]:
    if rets_window is None or rets_window.empty or not names:
        return {t: 0.0 for t in names}
    sub = rets_window.loc[:, [c for c in names if c in rets_window.columns]].copy()
    vol = sub.std().replace(0, np.nan)
    inv = 1.0 / (vol + eps)
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = inv.sum()
    if s <= eps:
        return {t: 0.0 for t in names}
    w = inv / s
    return {t: float(w.get(t, 0.0)) for t in names}


def _cap_weights(names, cap_map: Optional[Dict[str, float]], eps: float = 1e-8) -> Dict[str, float]:
    if not names:
        return {}
    vals = pd.Series({t: float(cap_map.get(t, 0.0)) if cap_map else 0.0 for t in names})
    vals = vals.clip(lower=0.0)
    s = float(vals.sum())
    if s <= eps:
        # fallback to rank-proportional so it is not equal weight
        ranks = vals.rank(method="first").replace(0, 1.0)
        rsum = float(ranks.sum())
        return {t: float(ranks[t] / rsum) for t in names}
    w = vals / s
    return {t: float(w[t]) for t in names}


def _compute_sharpe_series(rets_window: pd.DataFrame, rf_annual: Optional[float]) -> pd.Series:
    """Return per-asset Sharpe using rets_window mean and std.
    rf_annual in decimal units (e.g., 0.05 for 5 percent).
    """
    if rets_window is None or rets_window.empty:
        return pd.Series(dtype="float64")
    mu = rets_window.mean()
    sig = rets_window.std().replace(0, np.nan)
    if rf_annual is None:
        rf_per = 0.0
    else:
        af = _periods_per_year(rets_window.index)
        rf_per = float(rf_annual) / af
    ex = mu - rf_per
    sharpe = ex / sig
    return sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _apply_sharpe_multipliers(sharpe: pd.Series) -> pd.Series:
    """Map Sharpe to a multiplier ladder:
       <0.5 -> 0.5x, 0.5-1.0 -> 1.0x, 1.0-2.0 -> 1.5x, >2.0 -> 2.0x
    """
    mult = pd.Series(1.0, index=sharpe.index, dtype="float64")
    mult[sharpe < 0.5] = 0.5
    mult[(sharpe >= 0.5) & (sharpe < 1.0)] = 1.0
    mult[(sharpe >= 1.0) & (sharpe < 2.0)] = 1.5
    mult[sharpe >= 2.0] = 2.0
    return mult


def _sharpe_weights(names, rets_window: pd.DataFrame, rf_annual: Optional[float]) -> Dict[str, float]:
    if rets_window is None or rets_window.empty or not names:
        return {t: 0.0 for t in names}
    sharpe_all = _compute_sharpe_series(rets_window, rf_annual)
    s = sharpe_all.reindex(names).fillna(0.0)
    # Exclude negative Sharpe from allocation
    s = s.clip(lower=0.0)
    mult = _apply_sharpe_multipliers(s)
    raw = s * mult
    total = float(raw.sum())
    if total <= 0.0:
        return {t: 0.0 for t in names}
    w = raw / total
    return {t: float(w.get(t, 0.0)) for t in names}


# ---------------------------- Constraints ---------------------------

def _enforce_caps(weights: Dict[str, float], sector_map: Dict[str, str],
                  max_w: float, max_sector_w: float) -> Dict[str, float]:
    if not weights:
        return weights

    # single-name cap
    w = {t: max(-max_w, min(max_w, float(wt))) for t, wt in weights.items()}

    # sector gross caps
    sector_gross = {}
    for t, wt in w.items():
        s = sector_map.get(t, "UNKNOWN")
        sector_gross[s] = sector_gross.get(s, 0.0) + abs(wt)

    for s, gross in sector_gross.items():
        if gross > max_sector_w and gross > 0.0:
            scale = max_sector_w / gross
            for t in list(w):
                if sector_map.get(t, "UNKNOWN") == s:
                    w[t] *= scale

    # renormalize long and short legs separately to preserve structure
    pos = sum(v for v in w.values() if v > 0)
    neg = -sum(v for v in w.values() if v < 0)

    if pos > 0:
        for t in list(w):
            if w[t] > 0:
                w[t] = w[t] / pos
    if neg > 0:
        for t in list(w):
            if w[t] < 0:
                w[t] = w[t] / neg
    return w


# --------------------------- Public API -----------------------------

def build_target_weights(
    sig_row: pd.Series,
    prices_window: Optional[pd.DataFrame],
    sector_map: Dict[str, str],
    cap_map: Optional[Dict[str, float]] = None,
    weighting: str = "vol",
    market_neutral: bool = False,
    max_w: float = 0.05,
    max_sector_w: float = 0.25,
    rf_annual: Optional[float] = None
) -> Dict[str, float]:
    """
    Convert a signal row (values in {-1, 0, 1}) into portfolio weights.

    Parameters
    ----------
    sig_row : pd.Series
        Signals at the rebalance date indexed by ticker: -1 short, 0 flat, +1 long.
    prices_window : pd.DataFrame
        Recent returns window (daily or monthly). Used for volatility and Sharpe.
        Columns are tickers, index is datetime-like.
    sector_map : dict
        Mapping {ticker: sector} used for sector caps.
    cap_map : dict, optional
        Mapping {ticker: market_cap} used when weighting="cap".
    weighting : {"vol", "cap", "sharpe"}
        Choice of scheme. No equal-weighting is provided.
    market_neutral : bool
        If True, target +1 gross for longs and -1 gross for shorts.
        If False, long-only using the long set only.
    max_w : float
        Max single-name absolute weight cap.
    max_sector_w : float
        Max sector gross weight cap.
    rf_annual : float, optional
        Annualized risk-free rate in decimal (e.g., 0.05). Used in Sharpe.

    Returns
    -------
    dict
        {ticker: target_weight}
    """
    # Identify active names
    longs = [normalize_ticker(t) for t, v in sig_row.items() if v == 1]
    shorts = [normalize_ticker(t) for t, v in sig_row.items() if v == -1]

    # Choose base weights per side
    if weighting == "vol":
        wl = _inverse_vol_weights(longs, prices_window) if longs else {}
        ws_abs = _inverse_vol_weights(shorts, prices_window) if shorts else {}
    elif weighting == "cap":
        wl = _cap_weights(longs, cap_map) if longs else {}
        ws_abs = _cap_weights(shorts, cap_map) if shorts else {}
    elif weighting == "sharpe":
        wl = _sharpe_weights(longs, prices_window, rf_annual) if longs else {}
        ws_abs = _sharpe_weights(shorts, prices_window, rf_annual) if shorts else {}
    else:
        raise ValueError("Unsupported weighting. Use one of: 'vol', 'cap', 'sharpe'.")

    # Combine long and short sides
    w = {}
    if market_neutral:
        # allocate 50 percent gross to longs and 50 percent gross to shorts
        for t, wt in wl.items():
            w[t] = w.get(t, 0.0) + 0.5 * wt
        for t, wt in ws_abs.items():
            w[t] = w.get(t, 0.0) - 0.5 * wt
    else:
        # long-only portfolio
        for t, wt in wl.items():
            w[t] = w.get(t, 0.0) + wt

    # Apply constraints
    w = _enforce_caps(w, sector_map, max_w=max_w, max_sector_w=max_sector_w)
    # Clean near-zero noise
    w = {t: float(wt) for t, wt in w.items() if abs(wt) > 1e-9}
    return w
