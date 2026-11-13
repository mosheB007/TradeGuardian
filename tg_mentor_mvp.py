#!/usr/bin/env python3
"""TradeGuardian - V0 technical learning layer (CLI).

An educational "mentor with a filter" for beginner swing traders (10-20 day
holds). The user enters a ticker; the script pulls ~6 months of daily prices
from Yahoo Finance, computes three indicators, classifies the stock into one
of three learning levels, and prints a Hebrew report telling the user what to
look for with their own eyes on an external TradingView chart.

This tool does NOT generate trading signals and does NOT trade.

Implements PRD sections 3.A (data), 3.B (indicators), 3.C (3-level filter),
3.D (earnings blackout + news veto), 3.E (capital shield / risk sizing) and
the chart-link requirement from section 4.

Flow:
    1. Fetch prices. If earnings are due today..+14 days -> red warning
       block, then the flow continues for learning purposes only.
       (Earnings date unknown -> yellow warning, continue.)
    2. Technical report + 3-level classification, followed by an
       educational valuation block (Trailing P/E, Forward P/E, PEG) that
       never affects the classification. RED -> stop.
    3. GREEN/YELLOW: show headlines, ask the user for a news veto.
       'כן' -> disqualified, stop.  'לא' -> capital shield block.

Capital shield (shown only for GREEN/YELLOW), based on ATR_14:
    stop-loss    = entry - 2 * ATR_14            (entry = last close)
    take-profit  = entry + 2 * (entry - stop)     (1:2 risk/reward)
    max shares   = floor(capital * 1% / (entry - stop)),
                   additionally capped so the position never costs more than
                   the virtual capital itself (no leverage).

Classification rules (evaluated on the last *completed* trading day - if the
exchange session is still open, today's partial bar is excluded because its
partial volume would distort the volume check):
    RED     close <= SMA_50                     (trend is not positive)
    GREEN   close >  SMA_50 and RSI_14 < 45 and volume < Volume_SMA_20
    YELLOW  close >  SMA_50 but RSI_14 >= 45 or volume >= Volume_SMA_20

Note: the PRD defines RED as "close < SMA_50" and GREEN/YELLOW as
"close > SMA_50". The tie (close == SMA_50) is treated as RED, the
conservative choice for a beginner tool: price is not *above* the trend line.

Usage:
    python tg_mentor_mvp.py

Requirements:
    pip install yfinance pandas pandas-ta   (pandas-ta needs Python >= 3.12)
"""

from __future__ import annotations

import io
import logging
import math
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from pandas_ta.momentum import rsi
from pandas_ta.overlap import sma
from pandas_ta.volatility import atr
from yfinance.exceptions import (
    YFPricesMissingError,  # noqa: F401 - subclass of YFTickerMissingError
    YFRateLimitError,
    YFTickerMissingError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HISTORY_PERIOD = "6mo"
HISTORY_INTERVAL = "1d"

SMA_LENGTH = 50
RSI_LENGTH = 14
VOLUME_SMA_LENGTH = 20
RSI_THRESHOLD = 45.0
ATR_LENGTH = 14

# Capital shield (PRD 3.E) - a simulated account, never real money.
VIRTUAL_CAPITAL = 10_000.0  # USD
RISK_PER_TRADE = 0.01       # max 1% of capital lost if the stop-loss is hit
STOP_ATR_MULTIPLIER = 2.0   # stop-loss distance = 2 x ATR below entry
REWARD_RISK_RATIO = 2.0     # take-profit distance = 2 x risk (1:2)

# Entry planning (GUI) - reference levels, never a trading signal.
SUPPORT_LOOKBACK = 20            # support = lowest low of the previous 20 sessions
KNIFE_DAY_DROP_ATR = 1.5         # one-day drop >= 1.5 x ATR ...
KNIFE_DAY_VOLUME_RATIO = 1.5     # ... on volume >= 1.5 x its 20-day average
KNIFE_SLIDE_DAYS = 5             # or, over the last 5 sessions,
KNIFE_SLIDE_ATR = 3.0            # a decline >= 3 x ATR
# ATR balloons during a crash, which makes the two ATR-based tests above harder
# to trip exactly when they matter, so distance from SMA_50 is tested in plain
# percent. Calibrated on 56 US tickers, 2017-2026 (see PRD 3.G): at 15% or more
# below SMA_50 the stock fell a further 10%+ within 10 sessions in ~37% of
# days (~33% on the first day of an episode) vs ~12% of all days.
KNIFE_SMA_DISTANCE = 0.15        # close 15%+ below SMA_50
ENTRY_PULLBACK_ATR = 1.0         # normal limit: at most 1 x ATR below the close
ENTRY_KNIFE_DISCOUNT_ATR = 2.0   # falling knife: 2 x ATR below the close

# Historical statistics for the GUI "situation picture" (PRD 3.H).
STATS_RANGE_DAYS = 10            # horizon of the expected-range bands
STATS_RANGE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
STATS_FILL_DAYS = 10             # a limit order is given 10 sessions to fill
STATS_LIMIT_DEPTHS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # ATRs below the close
STATS_TRADE_DAYS = 30            # horizon for "target or stop first"
STATS_UP_DAYS = 20               # horizon of the plain "was it higher" rate
STATS_MIN_DAYS = 750             # ~3 years: below this, no per-stock statistics
STATS_MIN_STATE_DAYS = 200       # per-state trade odds need this many days
LADDER_DEPTHS = (0.5, 1.0, 1.5, 2.0, 3.0)  # limit-ladder steps shown, in ATRs
TREND_SLOPE_DAYS = 20            # SMA_50 "rising" = above its value 20 sessions ago

# News & warnings (PRD 3.D).
EARNINGS_BLACKOUT_DAYS = 14  # earnings today..14 days ahead -> disqualify
NEWS_LIMIT = 5               # max headlines shown
MAX_VETO_ATTEMPTS = 3        # unclear answers before disqualifying for safety
YES_ANSWERS = frozenset({"כן", "yes", "y"})
NO_ANSWERS = frozenset({"לא", "no", "n"})
VETO_PROMPT = (
    "האם זיהית בחדשות אירוע חריג שדורש פסילה? "
    "(הקלד 'כן' כדי לפסול, או 'לא' כדי להמשיך לניהול הסיכונים): "
)
# Valuation context (educational, never affects the technical Level).
PEG_ATTRACTIVE_BELOW = 1.0  # 0 < PEG < 1.0 -> cheap relative to growth
PEG_FAIR_MAX = 2.0          # 1.0 <= PEG <= 2.0 -> fair; above -> premium

EARNINGS_UNKNOWN_MESSAGE = (
    "לא אותר תאריך דוח כספי. באחריותך לבדוק זאת ידנית לפני כניסה לעסקה."
)
YAHOO_NEWS_URL = "https://finance.yahoo.com/quote/{symbol}/news"  # manual fallback
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
RSS_TIMEOUT_SECONDS = 5
RSS_MAX_BYTES = 1_000_000  # a headline feed is ~10 KB; cap against abuse
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

REQUIRED_COLUMNS = ("High", "Low", "Close", "Volume")

# Exchange-local time after which today's daily bar is treated as final.
# Covers NYSE/Nasdaq (16:00) and LSE (16:30) closes with a safety buffer.
MARKET_CLOSE_CUTOFF = time(17, 30)

# Letters, digits and the punctuation Yahoo uses in symbols
# (e.g. BRK-B, TEVA.TA, ^GSPC, EURUSD=X).
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")

TRADINGVIEW_CHART_URL = "https://www.tradingview.com/chart/?symbol={symbol}"

SEPARATOR = "=" * 64
THIN_SEPARATOR = "-" * 64


# ---------------------------------------------------------------------------
# Errors - every message is user-facing Hebrew text
# ---------------------------------------------------------------------------

class TradeGuardianError(Exception):
    """Base class for expected, user-facing failures (message is Hebrew)."""


class TickerInputError(TradeGuardianError):
    """The ticker typed by the user is empty or malformed."""


class DataFetchError(TradeGuardianError):
    """Price data could not be retrieved or is unusable for analysis."""


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class Level(Enum):
    """The three learning levels defined in PRD section 3.C."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class Snapshot:
    """Indicator values for the last trading day."""

    date: date
    close: float
    sma_50: float
    rsi_14: float
    volume: float
    volume_sma_20: float
    atr_14: float
    prev_close: float | None = None  # previous trading day's close
    # Entry planning (optional so older callers keep working):
    low_20: float | None = None           # lowest low of the previous 20 sessions
    close_slide_ago: float | None = None  # close KNIFE_SLIDE_DAYS sessions ago
    # True when `close` comes from today's still-open session (GUI live mode);
    # the volume fields then describe the last completed session.
    partial_session: bool = False
    sma_50_prev: float | None = None  # SMA_50 TREND_SLOPE_DAYS sessions ago

    @property
    def sma_rising(self) -> bool | None:
        """True if SMA_50 is above its value TREND_SLOPE_DAYS ago; None if unknown."""
        if self.sma_50_prev is None:
            return None
        return self.sma_50 > self.sma_50_prev

    @property
    def atr_pct(self) -> float:
        """Average daily range as a percent of the price."""
        return self.atr_14 / self.close * 100 if self.close else 0.0

    @property
    def daily_change(self) -> float | None:
        """Change vs the previous close, in price units; None if unknown."""
        if self.prev_close is None:
            return None
        return self.close - self.prev_close

    @property
    def daily_change_pct(self) -> float | None:
        """Change vs the previous close, in percent; None if unknown."""
        if self.prev_close is None or self.prev_close == 0:
            return None
        return (self.close / self.prev_close - 1) * 100

    @property
    def rsi_is_low(self) -> bool:
        return self.rsi_14 < RSI_THRESHOLD

    @property
    def volume_is_calm(self) -> bool:
        return self.volume < self.volume_sma_20

    @property
    def above_sma(self) -> bool:
        return self.close > self.sma_50


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def normalize_ticker(raw: str) -> str:
    """Trim/uppercase user input and validate it looks like a ticker symbol."""
    ticker = raw.strip().upper()
    if not ticker:
        raise TickerInputError("לא הוזן סימול מניה. יש להקליד טיקר, למשל: NVDA")
    if not TICKER_PATTERN.match(ticker):
        raise TickerInputError(
            f'הסימול "{raw.strip()}" אינו תקין. '
            "טיקר מכיל אותיות באנגלית וספרות בלבד, למשל: NVDA, AAPL, BRK-B"
        )
    return ticker


# ---------------------------------------------------------------------------
# Data fetching (PRD 3.A)
# ---------------------------------------------------------------------------

def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Download ~6 months of daily OHLCV data for `ticker` from Yahoo Finance.

    Raises:
        DataFetchError: on network failure, rate limiting, unknown ticker,
            empty/invalid payload, or any other API error. The message is
            Hebrew and safe to print directly to the user.
    """
    # yfinance hides errors by default and returns an empty frame instead,
    # which would make a network outage look like an unknown ticker.
    yf.config.debug.hide_exceptions = False

    try:
        history = yf.Ticker(ticker).history(
            period=HISTORY_PERIOD, interval=HISTORY_INTERVAL
        )
    except YFTickerMissingError as exc:  # also covers YFPricesMissingError
        raise _ticker_not_found(ticker) from exc
    except YFRateLimitError as exc:
        raise _rate_limited() from exc
    except OSError as exc:
        # yfinance's HTTP backend (curl_cffi) raises OSError subclasses for
        # both transport failures and HTTP error statuses; tell them apart.
        raise _classify_http_failure(ticker, exc) from exc
    except Exception as exc:  # noqa: BLE001 - any API failure must not crash
        raise _service_error(ticker) from exc

    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        raise _ticker_not_found(ticker)
    if any(col not in history.columns for col in REQUIRED_COLUMNS):
        raise DataFetchError(
            f"הנתונים שהתקבלו עבור {ticker} חסרים או פגומים (אין נתוני מחיר/מחזור)."
        )
    history = history.dropna(subset=["Close"])
    if history.empty:
        raise _ticker_not_found(ticker)
    return history


def _classify_http_failure(ticker: str, exc: OSError) -> DataFetchError:
    """Map an OSError from the HTTP layer to the right Hebrew message."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if not status:
        # Never reached the server (curl_cffi reports status 0 or no response).
        return _network_error()
    if status == 404:
        return _ticker_not_found(ticker)  # Yahoo's answer for unknown symbols
    if status == 429:
        return _rate_limited()
    return _service_error(ticker)


def _ticker_not_found(ticker: str) -> DataFetchError:
    return DataFetchError(
        f'לא נמצאו נתונים עבור הסימול "{ticker}". '
        "ודא שהטיקר נכון (למשל NVDA) ושהמניה עדיין נסחרת."
    )


def _rate_limited() -> DataFetchError:
    return DataFetchError(
        "שרת הנתונים (Yahoo Finance) חסם זמנית את הבקשה עקב ריבוי בקשות. "
        "המתן מספר דקות ונסה שוב."
    )


def _network_error() -> DataFetchError:
    return DataFetchError(
        "שגיאת רשת: לא ניתן להתחבר לשרת הנתונים (Yahoo Finance). "
        "בדוק את חיבור האינטרנט ונסה שוב."
    )


def _service_error(ticker: str) -> DataFetchError:
    return DataFetchError(
        f"אירעה שגיאה בעת שליפת הנתונים עבור {ticker} מ-Yahoo Finance. "
        "ייתכן שהשירות אינו זמין כרגע. נסה שוב מאוחר יותר."
    )


def _now(tz) -> pd.Timestamp:
    """Current time in `tz` (None = naive local time). Patched in tests."""
    return pd.Timestamp.now(tz=tz)


def drop_incomplete_session(
    history: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, bool]:
    """Drop today's bar if the exchange session is still open.

    While the market trades, Yahoo's daily bar for today is partial: its
    volume is only a fraction of a full day, which would make the
    "volume below Volume_SMA_20" check pass artificially. The comparison
    is done in the exchange's own timezone (taken from the data index).

    Returns:
        (history, dropped) - `dropped` is True if the last bar was removed.
    """
    tz = getattr(history.index, "tz", None)  # only DatetimeIndex carries a tz
    now = _now(tz) if now is None else now
    if tz is not None and now.tzinfo is not None:
        now = now.tz_convert(tz)

    last_bar = pd.Timestamp(history.index[-1])
    if last_bar.date() == now.date() and now.time() < MARKET_CLOSE_CUTOFF:
        return history.iloc[:-1], True
    return history, False


# ---------------------------------------------------------------------------
# Indicators (PRD 3.B)
# ---------------------------------------------------------------------------

def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `history` with SMA_50, RSI_14, Volume_SMA_20 and ATR_14."""
    df = history.copy()
    df["SMA_50"] = sma(df["Close"], length=SMA_LENGTH)
    df["RSI_14"] = rsi(df["Close"], length=RSI_LENGTH)
    df["Volume_SMA_20"] = sma(df["Volume"], length=VOLUME_SMA_LENGTH)
    df["ATR_14"] = atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)
    # Support for entry planning: today's bar is excluded (shift), so a calm
    # pullback that touches the level is not confused with the level itself.
    df["Low_20"] = df["Low"].rolling(SUPPORT_LOOKBACK).min().shift(1)
    return df


def _optional_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def latest_snapshot(df: pd.DataFrame) -> Snapshot:
    """Extract the last trading day's values; fail if any indicator is missing."""
    indicator_columns = (
        "Close", "SMA_50", "RSI_14", "Volume", "Volume_SMA_20", "ATR_14"
    )
    if df.empty or any(
        col not in df.columns or pd.isna(df[col].iloc[-1]) for col in indicator_columns
    ):
        raise DataFetchError(
            f"אין מספיק נתונים היסטוריים לחישוב האינדיקטורים "
            f"(נמצאו {len(df)} ימי מסחר; נדרשים לפחות {SMA_LENGTH}). "
            "ייתכן שמדובר במניה שהונפקה לאחרונה."
        )
    last = df.iloc[-1]
    return Snapshot(
        date=pd.Timestamp(df.index[-1]).date(),
        close=float(last["Close"]),
        sma_50=float(last["SMA_50"]),
        rsi_14=float(last["RSI_14"]),
        volume=float(last["Volume"]),
        volume_sma_20=float(last["Volume_SMA_20"]),
        atr_14=float(last["ATR_14"]),
        # The 6-month history already holds the previous session: no extra fetch.
        prev_close=float(df["Close"].iloc[-2]) if len(df) >= 2 else None,
        low_20=_optional_float(last["Low_20"]) if "Low_20" in df.columns else None,
        close_slide_ago=(
            float(df["Close"].iloc[-1 - KNIFE_SLIDE_DAYS])
            if len(df) > KNIFE_SLIDE_DAYS
            else None
        ),
        sma_50_prev=(
            _optional_float(df["SMA_50"].iloc[-1 - TREND_SLOPE_DAYS])
            if len(df) > TREND_SLOPE_DAYS
            else None
        ),
    )


# ---------------------------------------------------------------------------
# 3-level learning filter (PRD 3.C)
# ---------------------------------------------------------------------------

def classify(snapshot: Snapshot) -> Level:
    """Classify the last trading day into GREEN / YELLOW / RED."""
    if not snapshot.above_sma:
        return Level.RED
    if snapshot.rsi_is_low and snapshot.volume_is_calm:
        return Level.GREEN
    return Level.YELLOW


# ---------------------------------------------------------------------------
# News & warnings - Human-in-the-Loop (PRD 3.D)
#
# Yahoo's earnings/news endpoints are unreliable, so every fetch here is
# best-effort: any failure or odd payload degrades to "no data" and the CLI
# tells the user to check manually. These functions never raise.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsItem:
    """One headline, normalized from either yfinance news format."""

    title: str
    url: str | None
    publisher: str | None
    published: str | None  # ISO date (YYYY-MM-DD)


def _to_date(value: object) -> date | None:
    """Coerce a date-like calendar value to `date`; None if not possible."""
    if value is pd.NaT:
        return None
    if isinstance(value, datetime):  # includes pd.Timestamp
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def fetch_next_earnings_date(ticker: str, today: date | None = None) -> date | None:
    """Return the next earnings date on/after `today`, or None if unknown."""
    today = today or date.today()
    try:
        calendar = yf.Ticker(ticker).calendar
        if not isinstance(calendar, dict):
            return None
        raw = calendar.get("Earnings Date")
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        upcoming = [
            d for d in (_to_date(v) for v in values) if d is not None and d >= today
        ]
        return min(upcoming, default=None)
    except Exception:  # noqa: BLE001 - unstable endpoint must never crash the CLI
        return None


def is_earnings_too_close(earnings: date, today: date) -> bool:
    """True if earnings fall today or within the next EARNINGS_BLACKOUT_DAYS."""
    return 0 <= (earnings - today).days <= EARNINGS_BLACKOUT_DAYS


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _rss_url(ticker: str) -> str:
    query = urlencode({"s": ticker, "region": "US", "lang": "en-US"})
    return f"{YAHOO_RSS_URL}?{query}"


def _download_rss(url: str) -> bytes:
    """GET `url` with a browser User-Agent (Yahoo blocks the urllib default)."""
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=RSS_TIMEOUT_SECONDS) as response:
        return response.read(RSS_MAX_BYTES)


def _rss_date(raw: str | None) -> str | None:
    """RFC 822 pubDate ('Fri, 09 Oct 2026 19:47:53 +0000') -> 'YYYY-MM-DD'."""
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def _parse_rss(payload: bytes) -> list[NewsItem]:
    """Extract up to NEWS_LIMIT titled <item>s from an RSS 2.0 document.

    Raises ET.ParseError on malformed XML. ElementTree never fetches external
    entities, and the bundled expat rejects entity-expansion bombs.
    """
    root = ET.fromstring(payload)
    items: list[NewsItem] = []
    for node in root.iter("item"):
        title = _text(node.findtext("title"))
        if title is None:
            continue
        items.append(NewsItem(
            title=title,
            url=_text(node.findtext("link")),
            publisher=None,  # Yahoo's RSS carries no publisher field
            published=_rss_date(node.findtext("pubDate")),
        ))
        if len(items) == NEWS_LIMIT:
            break
    return items


def fetch_news(ticker: str) -> list[NewsItem]:
    """Return up to NEWS_LIMIT recent headlines from Yahoo's public RSS feed.

    Any network/HTTP/timeout/XML failure degrades to an empty list, which the
    report renders as a manual-check warning.
    """
    try:
        return _parse_rss(_download_rss(_rss_url(ticker)))
    except Exception:  # noqa: BLE001 - unstable endpoint must never crash the CLI
        return []


def ask_news_veto() -> bool:
    """Ask the user whether the news justify disqualifying the stock.

    Returns True to disqualify. After MAX_VETO_ATTEMPTS unclear answers the
    stock is disqualified for safety rather than looping forever.
    """
    for _ in range(MAX_VETO_ATTEMPTS):
        answer = input(VETO_PROMPT).strip().lower()
        if answer in YES_ANSWERS:
            return True
        if answer in NO_ANSWERS:
            return False
        print("  תשובה לא ברורה. יש להקליד 'כן' או 'לא'.")
    print("לא התקבלה תשובה ברורה - המניה נפסלת ליתר ביטחון.")
    return True


# ---------------------------------------------------------------------------
# Valuation context - educational only (P/E, Forward P/E, PEG)
#
# Independent of the technical Level: it never changes GREEN/YELLOW/RED and
# never stops the flow. Yahoo's `.info` endpoint is unreliable and often
# empty (ETFs, small caps), so the fetch is best-effort and never raises.
# ---------------------------------------------------------------------------

class ValuationTier(Enum):
    ATTRACTIVE = "attractive"      # מחיר טוב / זול יחסית לצמיחה: 0 < PEG < 1.0
    FAIR = "fair"                  # מחיר בינוני / הוגן: 1.0 <= PEG <= 2.0
    EXPENSIVE = "expensive"        # מחיר מנופח / פרמיה: PEG > 2.0
    UNPROFITABLE = "unprofitable"  # חברה מפסידה / ללא רווח נקי
    UNKNOWN = "unknown"            # נתוני תמחור חסרים (תעודת סל, אין נתוני צמיחה)


@dataclass(frozen=True)
class Valuation:
    """Valuation multiples for one ticker; every field may be missing."""

    trailing_pe: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    trailing_eps: float | None = None
    forward_eps: float | None = None
    tier: ValuationTier = ValuationTier.UNKNOWN


def _clean_float(val: object) -> float | None:
    """Coerce a number-like value to a finite float; None if not possible.

    Rejects None, bool (a subclass of int), NaN and +/-inf. Numeric strings
    are accepted because Yahoo occasionally sends e.g. "Infinity" or "12.5".
    """
    if val is None or isinstance(val, bool):
        return None
    try:
        number = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def classify_valuation(
    trailing_pe: float | None,
    forward_pe: float | None,
    peg_ratio: float | None,
    trailing_eps: float | None,
    forward_eps: float | None = None,
) -> ValuationTier:
    """Map the multiples to a tier. PEG decides; P/E alone is never enough.

    Any sign of losses - past (trailing EPS) or expected (forward EPS or
    forward P/E) - wins over the PEG, so a company expected to lose money
    is never shown as attractive or fair.
    """
    if trailing_eps is not None and trailing_eps <= 0:
        return ValuationTier.UNPROFITABLE
    if (forward_eps is not None and forward_eps <= 0) or (
        forward_pe is not None and forward_pe <= 0
    ):
        return ValuationTier.UNPROFITABLE
    if peg_ratio is not None:
        if peg_ratio <= 0:  # negative earnings or negative growth
            return ValuationTier.UNPROFITABLE
        if peg_ratio < PEG_ATTRACTIVE_BELOW:
            return ValuationTier.ATTRACTIVE
        if peg_ratio <= PEG_FAIR_MAX:
            return ValuationTier.FAIR
        return ValuationTier.EXPENSIVE
    if trailing_pe is not None and trailing_pe <= 0:
        return ValuationTier.UNPROFITABLE
    # Without growth data we cannot tell whether a high P/E is justified.
    return ValuationTier.UNKNOWN


def fetch_valuation(ticker: str) -> Valuation:
    """Best-effort valuation multiples from Yahoo. Never raises."""
    try:
        return valuation_from_info(yf.Ticker(ticker).info)
    except Exception:  # noqa: BLE001 - unstable endpoint must never crash the CLI
        return Valuation()


def valuation_from_info(info: object) -> Valuation:
    """Build a Valuation from an already-fetched `.info` dict. Never raises."""
    try:
        if not isinstance(info, dict):
            return Valuation()
        trailing_pe = _clean_float(info.get("trailingPE"))
        forward_pe = _clean_float(info.get("forwardPE"))
        trailing_eps = _clean_float(info.get("trailingEps"))
        forward_eps = _clean_float(info.get("forwardEps"))
        # Explicit None check, not `or`: a reported PEG of 0.0 is falsy but
        # is real data (it classifies as UNPROFITABLE) and must be kept.
        trailing_peg = _clean_float(info.get("trailingPegRatio"))
        peg_ratio = (
            trailing_peg
            if trailing_peg is not None
            else _clean_float(info.get("pegRatio"))
        )
        if (
            peg_ratio is None
            and forward_pe is not None and forward_pe > 0
            and trailing_eps is not None and trailing_eps > 0
            and forward_eps is not None and forward_eps > trailing_eps
        ):
            # Fallback: PEG from the EPS growth implied by analyst estimates.
            growth_pct = ((forward_eps / trailing_eps) - 1.0) * 100.0
            if growth_pct > 0:
                peg_ratio = forward_pe / growth_pct
        tier = classify_valuation(
            trailing_pe, forward_pe, peg_ratio, trailing_eps, forward_eps
        )
        return Valuation(
            trailing_pe, forward_pe, peg_ratio, trailing_eps, forward_eps, tier
        )
    except Exception:  # noqa: BLE001 - unstable endpoint must never crash the CLI
        return Valuation()


# ---------------------------------------------------------------------------
# Live price layer (GUI) - best-effort, never raises
#
# Yahoo is an unofficial source: the price may lag by up to ~15 minutes and
# the endpoint may be missing fields or rate-limited. Everything here
# degrades to "no quote" rather than failing the analysis.
# ---------------------------------------------------------------------------

SESSION_PRE, SESSION_REGULAR, SESSION_POST = "pre", "regular", "post"
SESSION_CLOSED, SESSION_UNKNOWN = "closed", "unknown"


@dataclass(frozen=True)
class LiveQuote:
    """The most recent price Yahoo reports, including extended hours."""

    price: float
    session: str                           # one of the SESSION_* constants
    time: datetime | None = None           # timezone-aware (UTC); None if unknown
    reference_close: float | None = None   # last official close to compare with

    @property
    def change(self) -> float | None:
        if self.reference_close is None:
            return None
        return self.price - self.reference_close

    @property
    def change_pct(self) -> float | None:
        if not self.reference_close:
            return None
        return (self.price / self.reference_close - 1) * 100


def fetch_ticker_info(ticker: str) -> dict:
    """Yahoo's `.info` dict, or {} on any failure. Never raises."""
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:  # noqa: BLE001 - unstable endpoint must never crash the app
        return {}


def _positive(value: object) -> float | None:
    number = _clean_float(value)
    return number if number is not None and number > 0 else None


def _epoch_to_datetime(value: object) -> datetime | None:
    """Yahoo timestamps are epoch seconds; anything else -> None."""
    seconds = _positive(value)
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _quote_from_info(info: dict) -> LiveQuote | None:
    state = str(info.get("marketState") or "").upper()
    regular = _positive(info.get("regularMarketPrice")) or _positive(
        info.get("currentPrice")
    )
    pre = _positive(info.get("preMarketPrice"))
    post = _positive(info.get("postMarketPrice"))
    regular_time = _epoch_to_datetime(info.get("regularMarketTime"))

    if state.startswith("PRE") and state != "PREPRE" and pre is not None:
        # Pre-market: compare with yesterday's official close.
        return LiveQuote(
            pre, SESSION_PRE, _epoch_to_datetime(info.get("preMarketTime")), regular
        )
    if state in ("POST", "POSTPOST", "PREPRE", "CLOSED") and post is not None:
        # After-hours (or overnight): the last extended-hours trade.
        return LiveQuote(
            post, SESSION_POST, _epoch_to_datetime(info.get("postMarketTime")), regular
        )
    if regular is None:
        return None
    if state == "REGULAR":
        previous = _positive(info.get("regularMarketPreviousClose")) or _positive(
            info.get("previousClose")
        )
        return LiveQuote(regular, SESSION_REGULAR, regular_time, previous)
    # Closed (or an unknown state) with no extended-hours price: the last
    # official close is all there is, so there is nothing to compare it with.
    session = SESSION_UNKNOWN if not state else SESSION_CLOSED
    return LiveQuote(regular, session, regular_time, None)


def _quote_from_minute_bars(ticker: str) -> LiveQuote | None:
    """Fallback: the last 1-minute bar, extended hours included."""
    bars = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True)
    if not isinstance(bars, pd.DataFrame) or bars.empty or "Close" not in bars.columns:
        return None
    closes = bars["Close"].dropna()
    if closes.empty:
        return None
    price = _positive(closes.iloc[-1])
    if price is None:
        return None
    stamp = pd.Timestamp(closes.index[-1])
    when = stamp.tz_convert("UTC").to_pydatetime() if stamp.tzinfo else None
    return LiveQuote(price, SESSION_UNKNOWN, when, None)


def fetch_live_quote(ticker: str, info: dict | None = None) -> LiveQuote | None:
    """Latest price from Yahoo, or None if unavailable. Never raises.

    Pass `info` (from fetch_ticker_info) to avoid a second request.
    """
    try:
        if info is None:
            info = fetch_ticker_info(ticker)
        quote = _quote_from_info(info) if isinstance(info, dict) else None
        return quote if quote is not None else _quote_from_minute_bars(ticker)
    except Exception:  # noqa: BLE001 - a missing quote must never stop the analysis
        return None


def build_live_snapshot(
    history: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[Snapshot, bool]:
    """Snapshot that includes today's partial bar while the session is open.

    Returns (snapshot, session_open). While the market trades, SMA_50, RSI_14
    and ATR_14 are computed on the partial bar, but the volume fields come
    from the last *completed* session: a partial day's volume is always low
    and would fake the "calm volume" condition. Outside trading hours this
    is exactly latest_snapshot() on the full history.
    """
    settled, session_open = drop_incomplete_session(history, now)
    if not session_open:
        return latest_snapshot(compute_indicators(history)), False
    live = latest_snapshot(compute_indicators(history))
    base = latest_snapshot(compute_indicators(settled))
    snapshot = replace(
        live,
        volume=base.volume,
        volume_sma_20=base.volume_sma_20,
        partial_session=True,
    )
    return snapshot, True


# ---------------------------------------------------------------------------
# Historical statistics - "what usually happened next" (GUI situation picture)
#
# One vectorised pass over a daily history. The app runs it live on the
# checked stock's full history, and research/calibrate.py runs the very same
# code on a basket to produce REFERENCE_STATS, so both use one definition.
# These are historical frequencies - not forecasts, and never a signal.
# ---------------------------------------------------------------------------

STATE_ABOVE, STATE_BELOW, STATE_KNIFE = "above", "below", "knife"
STOCK_STATES = (STATE_ABOVE, STATE_BELOW, STATE_KNIFE)


@dataclass(frozen=True)
class TradeOdds:
    """Outcome shares of a 2xATR-stop / 1:2-target trade within the horizon."""

    target: float   # share of days on which the target was reached first
    stop: float     # share on which the stop was hit first (ties count as stop)
    neither: float  # share on which neither happened within STATS_TRADE_DAYS
    days: int       # sample size


@dataclass(frozen=True)
class HistoryStats:
    days: int                                  # trading days in the sample
    years: float
    range_atr: tuple[float, ...]               # STATS_RANGE_QUANTILES, in ATRs
    fill: tuple[tuple[float, float], ...]      # (depth in ATRs, fill probability)
    odds: tuple[tuple[str, TradeOdds | None], ...]  # per state; None = too few days
    up_rate: float                             # share of days higher 20 sessions on

    def odds_for(self, state: str) -> TradeOdds | None:
        return dict(self.odds).get(state)


def history_events(history: pd.DataFrame) -> pd.DataFrame:
    """Per-day forward outcomes for a daily OHLCV history (may be empty).

    For every day with indicators and STATS_TRADE_DAYS of future data:
    the stock state, the move after STATS_RANGE_DAYS in ATR units, whether a
    limit order at each depth would have filled within STATS_FILL_DAYS, and
    how a trade entered at the close (stop 2xATR, target 1:2) ended.
    """
    columns = ["state", "move", "up", "outcome"] + [f"fill_{d}" for d in STATS_LIMIT_DEPTHS]
    try:
        df = compute_indicators(history).dropna(
            subset=["Close", "High", "Low", "SMA_50", "ATR_14", "Volume_SMA_20"]
        )
    except Exception:  # noqa: BLE001 - odd payloads mean "no statistics"
        return pd.DataFrame(columns=columns)
    n = len(df)
    horizon = max(STATS_TRADE_DAYS, STATS_UP_DAYS, STATS_RANGE_DAYS, STATS_FILL_DAYS)
    usable = n - horizon
    if usable <= KNIFE_SLIDE_DAYS:
        return pd.DataFrame(columns=columns)

    close, atr_ = df["Close"].to_numpy(float), df["ATR_14"].to_numpy(float)
    low, high = df["Low"].to_numpy(float), df["High"].to_numpy(float)
    sma_, vol = df["SMA_50"].to_numpy(float), df["Volume"].to_numpy(float)
    vol_avg = df["Volume_SMA_20"].to_numpy(float)

    prev = np.r_[np.nan, close[:-1]]
    ago = np.r_[np.full(KNIFE_SLIDE_DAYS, np.nan), close[:-KNIFE_SLIDE_DAYS]]
    with np.errstate(invalid="ignore"):
        far = close <= sma_ * (1 - KNIFE_SMA_DISTANCE)
        panic = (prev - close >= KNIFE_DAY_DROP_ATR * atr_) & (
            vol >= KNIFE_DAY_VOLUME_RATIO * vol_avg
        )
        slide = ago - close >= KNIFE_SLIDE_ATR * atr_
    knife = far | panic | slide
    state = np.where(knife, STATE_KNIFE, np.where(close > sma_, STATE_ABOVE, STATE_BELOW))

    idx = np.arange(KNIFE_SLIDE_DAYS, usable)
    idx = idx[atr_[idx] > 0]
    if idx.size == 0:
        return pd.DataFrame(columns=columns)
    c, a = close[idx], atr_[idx]
    out = pd.DataFrame({"state": state[idx]}, index=df.index[idx])
    out["move"] = (close[idx + STATS_RANGE_DAYS] - c) / a
    out["up"] = close[idx + STATS_UP_DAYS] > c

    # Windows of the *following* sessions: row i covers days i+1 .. i+k.
    low_fill = sliding_window_view(low[1:], STATS_FILL_DAYS)[idx].min(axis=1)
    for depth in STATS_LIMIT_DEPTHS:
        out[f"fill_{depth}"] = low_fill <= c - depth * a

    lows = sliding_window_view(low[1:], STATS_TRADE_DAYS)[idx]
    highs = sliding_window_view(high[1:], STATS_TRADE_DAYS)[idx]
    stop_hit = lows <= (c - STOP_ATR_MULTIPLIER * a)[:, None]
    target_hit = highs >= (c + REWARD_RISK_RATIO * STOP_ATR_MULTIPLIER * a)[:, None]
    never = STATS_TRADE_DAYS + 1
    first_stop = np.where(stop_hit.any(axis=1), stop_hit.argmax(axis=1), never)
    first_target = np.where(target_hit.any(axis=1), target_hit.argmax(axis=1), never)
    out["outcome"] = np.where(
        first_stop <= first_target,
        np.where(first_stop < never, "stop", "neither"),
        "target",
    )
    return out[columns]


def summarize_events(events: pd.DataFrame) -> HistoryStats | None:
    """Turn history_events() rows into shares; None below STATS_MIN_DAYS."""
    days = len(events)
    if days < STATS_MIN_DAYS:
        return None
    odds = []
    for state in STOCK_STATES:
        rows = events.loc[events["state"] == state, "outcome"]
        if len(rows) < STATS_MIN_STATE_DAYS:
            odds.append((state, None))
            continue
        share = rows.value_counts(normalize=True)
        odds.append((state, TradeOdds(
            float(share.get("target", 0.0)), float(share.get("stop", 0.0)),
            float(share.get("neither", 0.0)), len(rows),
        )))
    return HistoryStats(
        days=days,
        years=days / 252,
        range_atr=tuple(float(q) for q in events["move"].quantile(list(STATS_RANGE_QUANTILES))),
        fill=tuple((d, float(events[f"fill_{d}"].mean())) for d in STATS_LIMIT_DEPTHS),
        odds=tuple(odds),
        up_rate=float(events["up"].mean()),
    )


def compute_history_stats(history: object) -> HistoryStats | None:
    """Statistics of one stock's own history; None if too short or unusable."""
    if not isinstance(history, pd.DataFrame) or history.empty:
        return None
    try:
        return summarize_events(history_events(history))
    except Exception:  # noqa: BLE001 - statistics are optional, never fatal
        return None


def fetch_full_history(ticker: str) -> pd.DataFrame | None:
    """Every daily bar Yahoo has for `ticker`; None on any failure. Never raises."""
    try:
        history = yf.Ticker(ticker).history(period="max", interval=HISTORY_INTERVAL)
        if isinstance(history, pd.DataFrame) and not history.empty:
            return history
    except Exception:  # noqa: BLE001 - optional data, must never stop the analysis
        pass
    return None


@dataclass(frozen=True)
class RangeForecast:
    """Where the close landed after STATS_RANGE_DAYS sessions, in price units."""

    low_80: float
    low_50: float
    median: float
    high_50: float
    high_80: float


def expected_range(snapshot: Snapshot, stats: HistoryStats) -> RangeForecast:
    """Scale the historical ATR-unit quantiles to this stock's price and ATR."""
    low_80, low_50, median, high_50, high_80 = (
        snapshot.close + q * snapshot.atr_14 for q in stats.range_atr
    )
    return RangeForecast(max(low_80, 0.0), max(low_50, 0.0), median, high_50, high_80)


def limit_fill_probability(stats: HistoryStats, depth_atr: float) -> float:
    """Share of days a limit `depth_atr` ATRs below the close filled.

    Linear interpolation between the measured depths; at or above the close
    it is 1.0, and beyond the deepest measured depth it stays at that value
    (an upper bound).
    """
    if depth_atr <= 0:
        return 1.0
    points = ((0.0, 1.0),) + stats.fill
    for (d0, p0), (d1, p1) in zip(points, points[1:]):
        if depth_atr <= d1:
            return p0 + (p1 - p0) * (depth_atr - d0) / (d1 - d0)
    return points[-1][1]


@dataclass(frozen=True)
class LadderStep:
    depth_atr: float
    price: float
    pct_below: float  # distance below the close, in percent (positive number)


def limit_ladder(snapshot: Snapshot) -> tuple[LadderStep, ...]:
    """Candidate limit prices below the close, one per LADDER_DEPTHS step."""
    steps = []
    for depth in LADDER_DEPTHS:
        price = snapshot.close - depth * snapshot.atr_14
        if price <= 0 or snapshot.close <= 0:
            continue
        steps.append(LadderStep(depth, round(price, 2), (1 - price / snapshot.close) * 100))
    return tuple(steps)


def stock_state(snapshot: Snapshot, falling_knife: bool) -> str:
    if falling_knife:
        return STATE_KNIFE
    return STATE_ABOVE if snapshot.above_sma else STATE_BELOW


@dataclass(frozen=True)
class MarketRegime:
    """S&P 500 (SPY) against its own SMA_50."""

    close: float
    sma_50: float
    above: bool
    rising: bool | None  # SMA_50 higher than TREND_SLOPE_DAYS sessions ago


def market_regime(history: object) -> MarketRegime | None:
    """Regime from a daily index/ETF history; None if unusable. Never raises."""
    try:
        snapshot = latest_snapshot(compute_indicators(history))
    except Exception:  # noqa: BLE001 - optional context
        return None
    return MarketRegime(
        snapshot.close, snapshot.sma_50, snapshot.above_sma, snapshot.sma_rising
    )


# Reference basket: 55 US stocks and ETFs, every daily bar Yahoo has for them
# (1962-2026, ~510,000 trading days). Regenerate with research/calibrate.py.
# The last-10-years slice is within ~1 point of these on every figure.
REFERENCE_STATS = HistoryStats(
    days=510132,
    years=65.0,
    range_atr=(-2.55, -1.15, 0.30, 1.74, 3.16),
    fill=((0.5, 0.766), (1.0, 0.588), (1.5, 0.439), (2.0, 0.319), (2.5, 0.227),
          (3.0, 0.160), (4.0, 0.078)),
    odds=(
        ("above", TradeOdds(0.316, 0.517, 0.166, 303644)),
        ("below", TradeOdds(0.305, 0.502, 0.194, 177292)),
        ("knife", TradeOdds(0.260, 0.486, 0.254, 29196)),
    ),
    up_rate=0.574,
)
REFERENCE_TICKERS = 55
REFERENCE_SPAN = "1962-2026"
# Bear markets are the exception the averages hide: in 2000-2002 and 2008-2009
# the 1:2 target was reached on only ~16-25% of days (vs ~31% overall), and in
# 2000-2002 only 49% of days were higher 20 sessions later.
BEAR_TARGET_RANGE = (0.16, 0.25)

RSI_OVERSOLD, RSI_OVERBOUGHT = 30.0, 70.0


def rsi_zone(rsi_value: float) -> str:
    """'oversold' / 'weak' / 'firm' / 'overbought' (boundaries 30, 45, 70)."""
    if rsi_value < RSI_OVERSOLD:
        return "oversold"
    if rsi_value < RSI_THRESHOLD:
        return "weak"
    if rsi_value <= RSI_OVERBOUGHT:
        return "firm"
    return "overbought"


# ---------------------------------------------------------------------------
# Capital shield - risk management for a simulated trade (PRD 3.E)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskPlan:
    """Position sizing for a hypothetical long entry at the last close."""

    capital: float
    risk_pct: float
    atr: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_per_share: float
    risk_budget: float
    shares: int
    capped_by_capital: bool
    is_valid: bool

    @property
    def position_value(self) -> float:
        return self.shares * self.entry

    @property
    def max_loss(self) -> float:
        return self.shares * self.risk_per_share

    @property
    def expected_profit(self) -> float:
        """Profit if the take-profit target is reached."""
        return self.shares * (self.take_profit - self.entry)


def compute_risk_plan(
    snapshot: Snapshot,
    capital: float = VIRTUAL_CAPITAL,
    risk_pct: float = RISK_PER_TRADE,
    entry: float | None = None,
) -> RiskPlan:
    """Compute stop-loss, 1:2 take-profit and max share count for `snapshot`.

    `entry` is the planned entry price; by default the last close.

    The share count is floored so the loss at the stop never exceeds
    `capital * risk_pct`, and is also capped at what `capital` can buy.
    A plan is invalid (0 shares) when ATR is zero or the stop is <= 0.
    """
    if entry is None:
        entry = snapshot.close
    stop_loss = entry - STOP_ATR_MULTIPLIER * snapshot.atr_14
    risk_per_share = entry - stop_loss
    take_profit = entry + REWARD_RISK_RATIO * risk_per_share
    risk_budget = capital * risk_pct

    is_valid = risk_per_share > 0 and stop_loss > 0 and entry > 0
    shares = 0
    capped = False
    if is_valid:
        shares = math.floor(risk_budget / risk_per_share)
        affordable = math.floor(capital / entry)
        if shares > affordable:
            shares, capped = affordable, True

    return RiskPlan(
        capital=capital,
        risk_pct=risk_pct,
        atr=snapshot.atr_14,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_per_share=risk_per_share,
        risk_budget=risk_budget,
        shares=shares,
        capped_by_capital=capped,
        is_valid=is_valid,
    )


# ---------------------------------------------------------------------------
# Entry planning - educational reference levels (GUI only)
#
# A fixed, transparent rule on daily prices: it does NOT read the news and
# does not predict anything. The limit price is a reference level for a
# learning plan, never a trading signal.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryCondition:
    """One checklist line. ok: True = met, False = not met, None = unknown."""

    label: str
    ok: bool | None


@dataclass(frozen=True)
class EntryPlan:
    conditions: tuple[EntryCondition, ...]
    falling_knife: bool           # panic day, sharp slide or far below SMA_50
    support: float | None         # nearest support level below the close
    limit_price: float            # suggested limit-order reference price
    far_below_sma: bool = False   # close KNIFE_SMA_DISTANCE or more below SMA_50


def _is_panic_day(s: Snapshot) -> bool | None:
    """Sharp one-day drop on unusually heavy volume; None if unknown."""
    if s.prev_close is None or s.atr_14 <= 0:
        return None
    sharp_drop = s.prev_close - s.close >= KNIFE_DAY_DROP_ATR * s.atr_14
    if s.partial_session:
        # Today's volume is not final yet, so the drop alone raises the flag
        # (deliberately conservative while the session is still open).
        return sharp_drop
    return sharp_drop and s.volume >= KNIFE_DAY_VOLUME_RATIO * s.volume_sma_20


def _is_sharp_slide(s: Snapshot) -> bool | None:
    """Close fell KNIFE_SLIDE_ATR x ATR or more in KNIFE_SLIDE_DAYS sessions."""
    if s.close_slide_ago is None or s.atr_14 <= 0:
        return None
    return s.close_slide_ago - s.close >= KNIFE_SLIDE_ATR * s.atr_14


def _is_far_below_sma(s: Snapshot) -> bool:
    """Close is KNIFE_SMA_DISTANCE (15%) or more below SMA_50."""
    return s.sma_50 > 0 and s.close <= s.sma_50 * (1 - KNIFE_SMA_DISTANCE)


def _negate(value: bool | None) -> bool | None:
    return None if value is None else not value


def compute_entry_plan(
    snapshot: Snapshot,
    earnings_blocked: bool | None = None,
    valuation_tier: ValuationTier = ValuationTier.UNKNOWN,
) -> EntryPlan:
    """Build the entry checklist and a rule-based limit-price reference.

    Limit price:
      * falling knife -> close - ENTRY_KNIFE_DISCOUNT_ATR x ATR (conditional
        on the price stabilizing first);
      * otherwise     -> the nearest support below the close (SMA_50 or the
        previous 20-day low), but no further than ENTRY_PULLBACK_ATR x ATR.
    """
    s = snapshot
    panic, slide = _is_panic_day(s), _is_sharp_slide(s)
    far_below = _is_far_below_sma(s)
    valuation_ok = {
        ValuationTier.ATTRACTIVE: True,
        ValuationTier.FAIR: True,
        ValuationTier.EXPENSIVE: False,
        ValuationTier.UNPROFITABLE: False,
    }.get(valuation_tier)
    conditions = (
        EntryCondition("מגמה חיובית: המחיר מעל ממוצע 50", s.above_sma),
        EntryCondition(
            "אין ירידת פאניקה: לא נרשמה ירידה יומית חדה במחזור מסחר חריג",
            _negate(panic),
        ),
        EntryCondition(
            f"אין צניחה מתמשכת: המחיר לא ירד בחדות ב-{KNIFE_SLIDE_DAYS} "
            "ימי המסחר האחרונים",
            _negate(slide),
        ),
        EntryCondition(
            f"המחיר אינו רחוק מהממוצע: פחות מ-{KNIFE_SMA_DISTANCE:.0%} "
            "מתחת לממוצע 50",
            not far_below,
        ),
        EntryCondition(
            f"אין דוח כספי ב-{EARNINGS_BLACKOUT_DAYS} הימים הקרובים",
            _negate(earnings_blocked),
        ),
        EntryCondition("התמחור אינו מנופח והחברה רווחית (לפי PEG)", valuation_ok),
    )
    falling_knife = panic is True or slide is True or far_below

    levels = [lvl for lvl in (s.sma_50, s.low_20) if lvl is not None and lvl < s.close]
    support = max(levels, default=None)
    if falling_knife:
        limit = s.close - ENTRY_KNIFE_DISCOUNT_ATR * s.atr_14
    else:
        limit = s.close - ENTRY_PULLBACK_ATR * s.atr_14
        if support is not None:
            limit = max(support, limit)
    if limit <= 0:  # degenerate data: never suggest a non-positive price
        limit = s.close
    return EntryPlan(conditions, falling_knife, support, round(limit, 2), far_below)


# ---------------------------------------------------------------------------
# Report rendering (Hebrew)
# ---------------------------------------------------------------------------

def tradingview_url(ticker: str) -> str:
    """Build a TradingView chart link (Yahoo 'BRK-B' -> TradingView 'BRK.B')."""
    return TRADINGVIEW_CHART_URL.format(symbol=ticker.replace("-", "."))


def _indicator_lines(s: Snapshot) -> list[str]:
    trend = "המחיר מעל הממוצע" if s.above_sma else "המחיר מתחת/על הממוצע"
    volume_pct = (s.volume / s.volume_sma_20 * 100) if s.volume_sma_20 else 0.0
    volume_note = "נמוך מהממוצע" if s.volume_is_calm else "גבוה מהממוצע"
    return [
        "נתונים טכניים:",
        f"  • מחיר סגירה אחרון:                    {s.close:,.2f}",
        f"  • ממוצע נע 50 יום (SMA_50):            {s.sma_50:,.2f}   ({trend})",
        f"  • מדד עוצמה יחסית RSI (14 ימים):       {s.rsi_14:.1f}",
        f"  • מחזור מסחר אחרון:                    {s.volume:,.0f}",
        f"  • ממוצע מחזור 20 יום (Volume_SMA_20):  {s.volume_sma_20:,.0f}"
        f"   ({volume_note}, {volume_pct:.0f}% מהממוצע)",
    ]


def _green_lines(s: Snapshot) -> list[str]:
    return [
        "סיווג: [ירוק] מתאימה להמשך בדיקה",
        "",
        "משמעות:",
        "  המגמה חיובית - המחיר נמצא מעל ממוצע 50.",
        f"  יש ירידה במחיר (RSI = {s.rsi_14:.1f}, מתחת ל-{RSI_THRESHOLD:.0f}),",
        "  ללא פאניקת מוכרים (מחזור המסחר האחרון נמוך מהממוצע).",
        "",
        "המשימה שלך: פתח את הגרף וחפש עצירת מחיר או נר היפוך.",
        "  מה לחפש בעיניים:",
        "  1. עצירת מחיר - האם הירידה נעצרת? חפש כמה נרות קטנים ברצף, או נגיעה",
        "     בקו ממוצע 50 / באזור תמיכה (שפל קודם) שממנו המחיר לא ממשיך לרדת.",
        "  2. נר היפוך - נר עם 'זנב' תחתון ארוך וגוף קטן (פטיש / Hammer),",
        "     או נר ירוק גדול ש'בולע' את הנר האדום שלפניו (Bullish Engulfing).",
        "  3. מחזור מסחר - ודא שבנרות הירידה המחזור נמוך (אין מכירה המונית),",
        "     ובנר ההיפוך המחזור עולה (קונים חוזרים לשוק).",
        "  כל עוד לא ראית עצירה או נר היפוך ברור - אין מה לעשות. סבלנות.",
    ]


def _yellow_lines(s: Snapshot) -> list[str]:
    reasons = []
    if not s.rsi_is_low:
        reasons.append(
            f"  - RSI = {s.rsi_14:.1f} ({RSI_THRESHOLD:.0f} ומעלה): המניה 'יקרה' יחסית -"
            " היא עדיין לא ירדה מספיק לנקודת כניסה נוחה."
        )
    if not s.volume_is_calm:
        reasons.append(
            "  - מחזור המסחר האחרון גבוה מהממוצע של 20 יום: יש תנודתיות / פעילות חריגה."
        )
    return [
        "סיווג: [צהוב] דורשת זהירות",
        "",
        "משמעות:",
        "  המגמה חיובית - המחיר נמצא מעל ממוצע 50,",
        "  אך התזמון אינו אידיאלי כרגע. הסיבה:",
        *reasons,
        "",
        "המשימה שלך: להמתין להזדמנות טובה יותר.",
        "  מה לחפש בעיניים:",
        "  1. האם המחיר 'מתוח' הרחק מעל קו ממוצע 50? אם כן - המתן לתיקון (ירידה)",
        "     לכיוון הממוצע לפני שבכלל שוקלים כניסה.",
        "  2. הסתכל על עמודות הווליום מתחת לגרף: האם יש עמודה חריגה בגובהה?",
        "     האם היא בנר אדום (לחץ מכירות) או ירוק (קנייה נלהבת)? בשני המקרים -",
        "     לא נכנסים ברגע של תנודתיות חריגה.",
        "  3. הוסף את המניה לרשימת מעקב ובדוק שוב בעוד כמה ימים - כאשר ה-RSI ירד",
        f"     מתחת ל-{RSI_THRESHOLD:.0f} והמחזור יירגע אל מתחת לממוצע.",
    ]


def _red_lines(s: Snapshot) -> list[str]:
    return [
        "סיווג: [אדום] נפסלת",
        "",
        "משמעות:",
        "  המגמה שלילית - המחיר נמצא מתחת לממוצע הנע 50 (או בדיוק עליו).",
        "",
        "המשימה שלך: אין לחפש עסקאות קנייה במניה זו כרגע. עדיף לשבת על הגדר.",
        "  פתח את הגרף לצורך לימוד בלבד, וחפש בעיניים:",
        "  1. ראה כיצד המחיר נמצא מתחת לקו ממוצע 50 - הקו משמש כעת 'תקרה' (התנגדות).",
        "  2. שים לב לרצף של שיאים נמוכים יותר ושפלים נמוכים יותר - סימן למגמת ירידה.",
        "  3. אל תנסה 'לתפוס סכין נופלת'. המניה תחזור לשיקול רק כשהמחיר יחזור",
        "     ויתבסס מעל ממוצע 50.",
    ]


_LEVEL_RENDERERS = {
    Level.GREEN: _green_lines,
    Level.YELLOW: _yellow_lines,
    Level.RED: _red_lines,
}


def _aligned(rows: list[tuple[str, str]]) -> list[str]:
    """Render label/value pairs with the values lined up in one column."""
    width = max(len(label) for label, _ in rows)
    return [f"  {label.ljust(width)}  {value}" for label, value in rows]


def _risk_lines(plan: RiskPlan) -> list[str]:
    pct = f"{plan.risk_pct:.0%}"
    lines = [
        "מגן הון (ניהול סיכונים לעסקה מדומה)",
        THIN_SEPARATOR,
    ]
    if not plan.is_valid:
        lines += _aligned([
            ("הון וירטואלי:", f"${plan.capital:,.2f}"),
            ("תנודתיות יומית ממוצעת (ATR):", f"{plan.atr:,.2f}"),
        ])
        lines.append(
            "  לא ניתן לחשב תוכנית סיכון: ה-ATR הוא אפס או שמחיר הסטופ-לוס יוצא שלילי."
        )
    else:
        lines += _aligned([
            ("הון וירטואלי:", f"${plan.capital:,.2f}"),
            (f"סיכון מקסימלי לעסקה ({pct}):", f"${plan.risk_budget:,.2f}"),
            ("תנודתיות יומית ממוצעת (ATR):", f"{plan.atr:,.2f}"),
            ("מחיר כניסה (סגירה אחרונה):", f"{plan.entry:,.2f}"),
            (
                f"קו הגנה (Stop-Loss) - מרווח של פי {STOP_ATR_MULTIPLIER:g} מהתנודתיות:",
                f"{plan.stop_loss:,.2f}",
            ),
            ("הפסד אפשרי על כל מניה בודדת:", f"{plan.risk_per_share:,.2f}"),
            (
                f"יעד רווח (Take-Profit), יחס 1:{REWARD_RISK_RATIO:g}:",
                f"{plan.take_profit:,.2f}",
            ),
            ("כמות מניות מקסימלית לקנייה:", f"{plan.shares:,}"),
            ("עלות הפוזיציה:", f"${plan.position_value:,.2f}"),
            ("הפסד מתוכנן אם הסטופ יופעל:", f"${plan.max_loss:,.2f}"),
            ("רווח מתוכנן אם היעד יושג:", f"${plan.expected_profit:,.2f}"),
        ])
        if plan.capped_by_capital:
            lines.append(
                "  * הכמות הוגבלה לפי ההון הווירטואלי (ללא מינוף), "
                f"ולכן הסיכון בפועל נמוך מ-{pct}."
            )
        elif plan.shares == 0:
            lines.append(
                "  * ההפסד האפשרי על כל מניה בודדת גבוה מהתקציב בסיכון - "
                f"לא ניתן לקנות אפילו מניה אחת בלי לחרוג מ-{pct}."
            )
    lines += [
        "",
        "  שים לב: זהו תכנון מתמטי לימודי. במקרה של פער מחירים (Gap) בפתיחת המסחר, "
        f"ההפסד בפועל עלול להיות גדול מ-{pct}",
        THIN_SEPARATOR,
    ]
    return lines


def _tradingview_setup_lines() -> list[str]:
    return [
        "הוראות הגדרה ב-TradingView:",
        "  1. לחץ למעלה על כפתור 'Indicators' (סמל של fx).",
        "  2. חפש 'SMA' (Moving Average Simple), לחץ עליו, כנס להגדרות שלו",
        "     (גלגל שיניים) ושנה את ה-Length ל-50.",
        "  3. חפש 'RSI' (Relative Strength Index) ולחץ עליו.",
        "  כעת אתה אמור לראות את קו הממוצע על הגרף,",
        "  ואת ה-RSI והמחזורים (Volume) בתחתית המסך.",
    ]


def build_report(ticker: str, snapshot: Snapshot, level: Level) -> str:
    """Render the Hebrew technical learning report (indicators, level, chart)."""
    lines = [
        SEPARATOR,
        f"TradeGuardian | דוח לימודי טכני עבור: {ticker}",
        SEPARATOR,
        f"יום המסחר האחרון בנתונים: {snapshot.date.isoformat()}",
        "",
        *_indicator_lines(snapshot),
        "",
        THIN_SEPARATOR,
        *_LEVEL_RENDERERS[level](snapshot),
        THIN_SEPARATOR,
        "",
        f"קישור לגרף המניה ב-TradingView: {tradingview_url(ticker)}",
        "",
        *_tradingview_setup_lines(),
    ]
    return "\n".join(lines)


def build_risk_block(snapshot: Snapshot) -> str:
    """Render the capital-shield block for a simulated entry at the last close."""
    return "\n".join(_risk_lines(compute_risk_plan(snapshot)))


def _days_until_text(days: int) -> str:
    if days == 0:
        return "היום"
    if days == 1:
        return "מחר"
    return f"בעוד {days} ימים"


def build_earnings_block(ticker: str, earnings: date, today: date) -> str:
    """Render the red 'earnings too close' warning (the flow continues after it)."""
    when = _days_until_text((earnings - today).days)
    return "\n".join([
        SEPARATOR,
        f"[אדום] נפסלת - דוח כספי קרוב: {ticker}",
        SEPARATOR,
        f"  החברה צפויה לפרסם דוח כספי בתאריך {earnings.isoformat()} ({when}).",
        f"  זה בתוך חלון של {EARNINGS_BLACKOUT_DAYS} הימים הקרובים - קרוב מדי ומסוכן לסחור.",
        "  סביב פרסום דוח כספי המחיר עלול לזנק או לצנוח בפער (Gap) חד בבת אחת,",
        "  וקו ההגנה (Stop-Loss) לא יוכל להגן עליך.",
        "  הניתוח ממשיך לצורכי לימוד בלבד. מומלץ להמתין לפרסום הדוח לפני כניסה לעסקה.",
    ])


def build_earnings_info_line(earnings: date, today: date) -> str:
    when = _days_until_text((earnings - today).days)
    return (
        f"תאריך הדוח הכספי הבא: {earnings.isoformat()} ({when}) - "
        f"מחוץ לחלון הסיכון של {EARNINGS_BLACKOUT_DAYS} ימים."
    )


VALUATION_TITLE = "בדיקת תמחור וערך (מכפילי רווח וצמיחה)"
NOT_AVAILABLE = "לא זמין"
VALUATION_VERDICTS = {
    ValuationTier.ATTRACTIVE: (
        "[ירוק] תמחור נוח (זול יחסית לצמיחה) - PEG מתחת ל-1.0: "
        "קצב הצמיחה של החברה מצדיק ואף עוקף את מכפיל הרווח שלה."
    ),
    ValuationTier.FAIR: (
        "[צהוב] תמחור בינוני (הוגן) - PEG בין 1.0 ל-2.0: "
        "המניה נסחרת סביב השווי ההוגן שלה ביחס לקצב הצמיחה."
    ),
    ValuationTier.EXPENSIVE: (
        "[אדום] תמחור מנופח (פרמיה) - PEG מעל 2.0: "
        "אתם משלמים פרמיה גבוהה על החברה ביחס לקצב הצמיחה שלה."
    ),
    ValuationTier.UNPROFITABLE: (
        "[אדום] חברה מפסידה או בצמיחה שלילית - אין מכפיל רווח חיובי תקין."
    ),
    ValuationTier.UNKNOWN: (
        "[צהוב] נתוני תמחור חסרים - לא ניתן לקבוע האם המחיר מנופח "
        "(נפוץ בתעודות סל או כשאין תחזיות אנליסטים)."
    ),
}


def _ratio_text(value: float | None) -> str:
    return NOT_AVAILABLE if value is None else f"{value:,.2f}"


def format_eps(eps: float) -> str:
    """Signed dollar amount with the minus before the '$': -1.2 -> '-$1.20'.

    A loss that rounds to zero is shown as '$0.00', never '-$0.00'.
    """
    if round(eps, 2) < 0:
        return f"-${abs(eps):.2f}"
    return f"${abs(eps):.2f}"


def _eps_suffix(eps: float | None) -> str:
    return "" if eps is None else f" (EPS = {format_eps(eps)})"


def build_valuation_block(v: Valuation) -> str:
    """Render the educational valuation section (independent of the Level)."""
    lines = [
        VALUATION_TITLE,
        THIN_SEPARATOR,
        f"  1. מכפיל רווח נוכחי (Trailing P/E): {_ratio_text(v.trailing_pe)}",
        "     מחיר המניה חלקי הרווח למניה ב-12 החודשים האחרונים"
        f"{_eps_suffix(v.trailing_eps)}.",
        f"  2. מכפיל רווח עתידי (Forward P/E): {_ratio_text(v.forward_pe)}",
        "     מחיר המניה חלקי הרווח למניה הצפוי בשנה הבאה"
        f"{_eps_suffix(v.forward_eps)}.",
    ]
    if (
        v.trailing_pe is not None
        and v.forward_pe is not None
        and v.forward_pe < v.trailing_pe
    ):
        lines.append("     הרווח צפוי לצמוח ולכן המכפיל העתידי נמוך יותר.")
    lines += [
        f"  3. יחס צמיחה (PEG Ratio): {_ratio_text(v.peg_ratio)}",
        "     מתאים את מכפיל הרווח לקצב הצמיחה של החברה (מכפיל רווח חלקי צמיחה באחוזים).",
        "",
        f"  {VALUATION_VERDICTS[v.tier]}",
        "  הערה: בדיקת התמחור היא הקשר לימודי בלבד ואינה משנה את הסיווג הטכני.",
        THIN_SEPARATOR,
    ]
    return "\n".join(lines)


def build_news_block(ticker: str, items: list[NewsItem]) -> str:
    """Render recent headlines for the user's manual veto decision."""
    lines = [
        "חדשות ואזהרות (Human-in-the-Loop)",
        THIN_SEPARATOR,
    ]
    if not items:
        lines += [
            f"  [צהוב] לא נמצאו כותרות חדשות עבור {ticker} (ייתכן שהשירות אינו זמין כרגע).",
            f"  בדוק ידנית: {YAHOO_NEWS_URL.format(symbol=ticker)}",
        ]
    for i, item in enumerate(items, start=1):
        lines.append(f"  {i}. {item.title}")
        source = " | ".join(part for part in (item.publisher, item.published) if part)
        if source:
            lines.append(f"     {source}")
        if item.url:
            lines.append(f"     {item.url}")
    lines += [
        "",
        "  עבור על הכותרות (ופתח קישורים לפי הצורך) וחפש אירוע חריג, למשל:",
        "  פשיטת רגל, חקירה רשמית, הנפקה מדללת, עזיבה פתאומית של מנכ\"ל, תביעה מהותית.",
        "  חדשות רגילות (דירוג אנליסטים, השקת מוצר) אינן סיבה לפסילה.",
        THIN_SEPARATOR,
    ]
    return "\n".join(lines)


def build_veto_block(ticker: str) -> str:
    return "\n".join([
        f"[אדום] נפסלת - {ticker} נפסלה לפי החלטתך על בסיס החדשות.",
        "  מגן ההון לא יחושב. זו בדיוק המטרה של זכות הווטו:",
        "  עדיף לפספס עסקה מאשר להיכנס לאירוע לא צפוי.",
    ])


def build_footer() -> str:
    return "\n".join([
        "הבהרה: המערכת היא כלי לימודי בלבד ואינה מספקת איתותי מסחר או ייעוץ השקעות.",
        "       ההחלטה והאחריות הן שלך בלבד.",
        SEPARATOR,
    ])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_error(message: str) -> None:
    print(f"\nשגיאה: {message}")
    print("הבדיקה הופסקה.")


def main() -> int:
    """Run one on-demand check. Returns a process exit code (0 = success)."""
    try:
        print("TradeGuardian - המורה הטכני למסחר סווינג (V0)")
        raw = input("הקלד סימול מניה (למשל NVDA): ")
        ticker = normalize_ticker(raw)

        print(f"\nשולף נתוני מחיר יומיים של חצי השנה האחרונה עבור {ticker}...")
        history = fetch_price_history(ticker)
        history, session_open = drop_incomplete_session(history)

        # PRD 3.D: an imminent earnings report gets a red warning block, but
        # the analysis continues behind it for learning purposes.
        today = date.today()
        earnings = fetch_next_earnings_date(ticker, today)
        earnings_blocked = earnings is not None and is_earnings_too_close(earnings, today)
        if earnings_blocked:
            print()
            print(build_earnings_block(ticker, earnings, today))

        snapshot = latest_snapshot(compute_indicators(history))
        level = classify(snapshot)

        if session_open:
            print(
                "הערה: המסחר היום עדיין פתוח והנתונים של היום חלקיים, "
                "ולכן הניתוח מבוסס על יום המסחר המלא האחרון."
            )
        if earnings is None:
            print(f"[צהוב] {EARNINGS_UNKNOWN_MESSAGE}")
        elif not earnings_blocked:  # the red block above already covers it
            print(build_earnings_info_line(earnings, today))
        print()
        print(build_report(ticker, snapshot, level))
        print()
        # Educational context only: shown for every level, never changes it.
        print(build_valuation_block(fetch_valuation(ticker)))
        print()

        if level is not Level.RED:
            # PRD 3.D Human-in-the-Loop: the user may veto based on the news,
            # and only then do we proceed to the capital shield (PRD 3.E).
            print(build_news_block(ticker, fetch_news(ticker)))
            vetoed = ask_news_veto()
            print()
            print(build_veto_block(ticker) if vetoed else build_risk_block(snapshot))
            print()

        print(build_footer())
        return 0
    except TradeGuardianError as exc:
        _print_error(str(exc))
    except (KeyboardInterrupt, EOFError):
        print("\n\nהפעולה בוטלה על ידי המשתמש. להתראות!")
    except Exception:  # noqa: BLE001 - last line of defense: never show a traceback
        _print_error("אירעה שגיאה לא צפויה בעיבוד הנתונים. נסה שוב או בדוק טיקר אחר.")
    return 1


def _configure_console() -> None:
    """Use UTF-8 for Hebrew I/O on Windows and silence yfinance's English logs.

    stdin matters when input is piped: Windows would otherwise decode it with
    the ANSI code page and garble Hebrew answers such as 'לא'.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if not isinstance(stream, io.TextIOWrapper):
            continue  # e.g. redirected to an object without reconfigure()
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


if __name__ == "__main__":
    _configure_console()
    sys.exit(main())
