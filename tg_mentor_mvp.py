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
    1. Fetch prices. If earnings are due today..+14 days -> RED, stop.
       (Earnings date unknown -> yellow warning, continue.)
    2. Technical report + 3-level classification. RED -> stop.
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
from dataclasses import dataclass
from datetime import date, datetime, time
from email.utils import parsedate_to_datetime
from enum import Enum
from urllib.parse import urlencode

import pandas as pd
import yfinance as yf
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
    prev_close: float | None = None  # previous trading day's close (display only)

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
    return df


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
) -> RiskPlan:
    """Compute stop-loss, 1:2 take-profit and max share count for `snapshot`.

    The share count is floored so the loss at the stop never exceeds
    `capital * risk_pct`, and is also capped at what `capital` can buy.
    A plan is invalid (0 shares) when ATR is zero or the stop is <= 0.
    """
    entry = snapshot.close
    stop_loss = entry - STOP_ATR_MULTIPLIER * snapshot.atr_14
    risk_per_share = entry - stop_loss
    take_profit = entry + REWARD_RISK_RATIO * risk_per_share
    risk_budget = capital * risk_pct

    is_valid = risk_per_share > 0 and stop_loss > 0
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
    """Render the red 'earnings too close' disqualification."""
    when = _days_until_text((earnings - today).days)
    return "\n".join([
        SEPARATOR,
        f"[אדום] נפסלת - דוח כספי קרוב: {ticker}",
        SEPARATOR,
        f"  החברה צפויה לפרסם דוח כספי בתאריך {earnings.isoformat()} ({when}).",
        f"  זה בתוך חלון של {EARNINGS_BLACKOUT_DAYS} הימים הקרובים - קרוב מדי ומסוכן לסחור.",
        "  סביב פרסום דוח כספי המחיר עלול לזנק או לצנוח בפער (Gap) חד בבת אחת,",
        "  וקו ההגנה (Stop-Loss) לא יוכל להגן עליך.",
        "  הניתוח נעצר כאן. בדוק את המניה שוב לאחר פרסום הדוח.",
    ])


def build_earnings_info_line(earnings: date, today: date) -> str:
    when = _days_until_text((earnings - today).days)
    return (
        f"תאריך הדוח הכספי הבא: {earnings.isoformat()} ({when}) - "
        f"מחוץ לחלון הסיכון של {EARNINGS_BLACKOUT_DAYS} ימים."
    )


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

        # PRD 3.D: an imminent earnings report disqualifies before any analysis.
        today = date.today()
        earnings = fetch_next_earnings_date(ticker, today)
        if earnings is not None and is_earnings_too_close(earnings, today):
            print()
            print(build_earnings_block(ticker, earnings, today))
            print(build_footer())
            return 0

        snapshot = latest_snapshot(compute_indicators(history))
        level = classify(snapshot)

        if session_open:
            print(
                "הערה: המסחר היום עדיין פתוח והנתונים של היום חלקיים, "
                "ולכן הניתוח מבוסס על יום המסחר המלא האחרון."
            )
        if earnings is None:
            print(f"[צהוב] {EARNINGS_UNKNOWN_MESSAGE}")
        else:
            print(build_earnings_info_line(earnings, today))
        print()
        print(build_report(ticker, snapshot, level))
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
