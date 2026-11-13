#!/usr/bin/env python3
"""Recompute TradeGuardian's reference statistics from Yahoo Finance history.

Downloads the full daily history of a basket of US stocks and ETFs and runs
the same code the app runs live on a single stock (tg_mentor_mvp.history_events
/ summarize_events), so the app's per-stock numbers and REFERENCE_STATS share
one definition.

Usage (from the repository root; needs network access to Yahoo Finance):

    python research/calibrate.py                  # download, then print tables
    python research/calibrate.py --cache DIR      # reuse/store downloads in DIR

The last block of output is a Python literal to paste over REFERENCE_STATS in
tg_mentor_mvp.py. Nothing in the app imports this file.

Caveats that apply to every number printed here:
  * survivorship bias - the basket only holds companies that still trade;
  * overlapping days are not independent observations;
  * no commissions, spreads or slippage;
  * frequencies describe the past and are not forecasts.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tg_mentor_mvp as tg  # noqa: E402

BASKET = """
AAPL MSFT NVDA AMZN GOOGL META TSLA AMD INTC NFLX CRM ORCL ADBE CSCO QCOM
JPM BAC GS WFC C V MA PYPL
KO PEP PG JNJ PFE MRK UNH WMT COST HD MCD NKE DIS
XOM CVX BA CAT GE F GM
PLTR COIN SNAP ROKU ZM PTON RIVN UBER SHOP DKNG
SPY QQQ IWM
""".split()

RECENT_YEARS = 10
CRISES = {"2000-2002": (2000, 2002), "2008-2009": (2008, 2009)}


def load(ticker: str, cache: Path) -> pd.DataFrame | None:
    path = cache / f"{ticker}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    history = tg.fetch_full_history(ticker)
    if history is not None:
        history.to_pickle(path)
    return history


def collect(cache: Path) -> pd.DataFrame:
    frames = []
    for ticker in BASKET:
        history = load(ticker, cache)
        if history is None:
            print(f"  skipped {ticker}: no data", file=sys.stderr)
            continue
        events = tg.history_events(history)
        if events.empty:
            print(f"  skipped {ticker}: history too short", file=sys.stderr)
            continue
        events = events.assign(ticker=ticker, year=events.index.year)
        frames.append(events.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def pct(value: float) -> str:
    return f"{100 * value:5.1f}%"


def report(name: str, events: pd.DataFrame) -> tg.HistoryStats | None:
    stats = tg.summarize_events(events)
    print(f"\n=== {name}: {len(events):,} days, {events['ticker'].nunique()} tickers ===")
    if stats is None:
        print("  too few days")
        return None
    quantiles = ", ".join(
        f"{int(q * 100)}%: {v:+.2f}" for q, v in zip(tg.STATS_RANGE_QUANTILES, stats.range_atr)
    )
    print(f"  higher after {tg.STATS_UP_DAYS} sessions: {pct(stats.up_rate)}")
    print(f"  close after {tg.STATS_RANGE_DAYS} sessions, in ATRs: {quantiles}")
    print("  limit fill within", tg.STATS_FILL_DAYS, "sessions:",
          ", ".join(f"{d:g} ATR {pct(p)}" for d, p in stats.fill))
    for state, odds in stats.odds:
        if odds is None:
            print(f"  {state:6s}: too few days")
            continue
        print(f"  {state:6s}: target {pct(odds.target)}  stop {pct(odds.stop)}  "
              f"neither {pct(odds.neither)}  ({odds.days:,} days, "
              f"{pct(odds.days / stats.days)} of all)")
    return stats


def literal(stats: tg.HistoryStats) -> str:
    odds = ",\n".join(
        f'        ("{state}", TradeOdds({o.target:.3f}, {o.stop:.3f}, {o.neither:.3f}, {o.days}))'
        for state, o in stats.odds if o is not None
    )
    fill = ", ".join(f"({d}, {p:.3f})" for d, p in stats.fill)
    rng = ", ".join(f"{v:.2f}" for v in stats.range_atr)
    return (
        "REFERENCE_STATS = HistoryStats(\n"
        f"    days={stats.days},\n    years=<span of the basket in years>,\n"
        f"    range_atr=({rng}),\n    fill=({fill}),\n"
        f"    odds=(\n{odds},\n    ),\n    up_rate={stats.up_rate:.3f},\n)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, help="directory for downloaded histories")
    args = parser.parse_args()
    cache = args.cache or Path(tempfile.mkdtemp(prefix="tg-calibrate-"))
    cache.mkdir(parents=True, exist_ok=True)

    events = collect(cache)
    last_year = int(events["year"].max())
    print(f"basket: {events['ticker'].nunique()} tickers, "
          f"{int(events['year'].min())}-{last_year}")
    full = report("full history", events)
    report(f"last {RECENT_YEARS} years", events[events["year"] > last_year - RECENT_YEARS])
    report(f"before the last {RECENT_YEARS} years",
           events[events["year"] <= last_year - RECENT_YEARS])
    for name, (start, end) in CRISES.items():
        report(f"crisis {name}", events[events["year"].between(start, end)])
    if full is not None:
        print("\n# Paste into tg_mentor_mvp.py:\n" + literal(full))
    return 0


if __name__ == "__main__":
    sys.exit(main())
