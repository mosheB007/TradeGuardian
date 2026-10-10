"""Tests for tg_mentor_mvp.py (TradeGuardian V0 technical learning layer).

All network access is mocked - these tests never call Yahoo Finance.
"""

import io
import sys
import urllib.error
from datetime import date, datetime, timedelta
from email.message import Message

import numpy as np
import pandas as pd
import pytest

import tg_mentor_mvp as tg

# Captured before the autouse fixture stubs it out, for TestDownloadRss.
ORIGINAL_DOWNLOAD_RSS = tg._download_rss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(closes, volumes=None):
    """Build a minimal daily OHLCV DataFrame like yfinance returns."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000] * n
    index = pd.bdate_range(end="2026-10-08", periods=n)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.asarray(volumes, dtype=float),
        },
        index=index,
    )


class FakeTicker:
    """Stand-in for yfinance.Ticker: preset history/calendar, or raising.

    Defaults model yfinance's common "no data" answer: no earnings date
    (calendar None).
    """

    def __init__(
        self,
        history_result=None,
        history_exc=None,
        calendar=None,
        calendar_exc=None,
    ):
        self._result = history_result
        self._exc = history_exc
        self._calendar = calendar
        self._calendar_exc = calendar_exc

    def history(self, *args, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._result

    @property
    def calendar(self):
        if self._calendar_exc is not None:
            raise self._calendar_exc
        return self._calendar


def days_from_today(n):
    return date.today() + timedelta(days=n)


def rss_item(
    title: str | None,
    link: str | None = "https://example.com/a",
    pub: str | None = "Fri, 09 Oct 2026 19:47:53 +0000",
) -> str:
    parts = []
    if title is not None:
        parts.append(f"<title>{title}</title>")
    if link is not None:
        parts.append(f"<link>{link}</link>")
    if pub is not None:
        parts.append(f"<pubDate>{pub}</pubDate>")
    return "<item>" + "".join(parts) + "</item>"


def rss_xml(*items):
    """Build a Yahoo-style RSS 2.0 document (bytes) from rss_item() strings."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<rss version="2.0"><channel>'
        "<title>Yahoo! Finance: NVDA News</title>"
        "<image><title>logo</title><link>http://finance.yahoo.com</link></image>"
        + "".join(items)
        + "</channel></rss>"
    ).encode("utf-8")


def patch_rss(monkeypatch, payload=None, exc=None):
    """Replace the RSS download with a canned payload or an exception."""
    calls = []

    def fake_download(url):
        calls.append(url)
        if exc is not None:
            raise exc
        return payload

    monkeypatch.setattr(tg, "_download_rss", fake_download)
    return calls


@pytest.fixture(autouse=True)
def no_network_news(monkeypatch):
    """Tests must never hit Yahoo: by default the RSS download fails."""
    patch_rss(monkeypatch, exc=OSError("network disabled in tests"))


def patch_ticker(monkeypatch, **kwargs):
    monkeypatch.setattr(tg.yf, "Ticker", lambda symbol: FakeTicker(**kwargs))


# ---------------------------------------------------------------------------
# Ticker normalization
# ---------------------------------------------------------------------------

class TestNormalizeTicker:
    def test_strips_and_uppercases(self):
        assert tg.normalize_ticker("  nvda ") == "NVDA"

    def test_allows_dash_and_dot(self):
        assert tg.normalize_ticker("brk-b") == "BRK-B"
        assert tg.normalize_ticker("teva.ta") == "TEVA.TA"

    def test_empty_raises_hebrew_error(self):
        with pytest.raises(tg.TickerInputError) as exc:
            tg.normalize_ticker("   ")
        assert "לא הוזן" in str(exc.value)

    def test_invalid_characters_raise_hebrew_error(self):
        with pytest.raises(tg.TickerInputError) as exc:
            tg.normalize_ticker("NV DA!")
        assert "אינו תקין" in str(exc.value)


# ---------------------------------------------------------------------------
# Data fetching (yfinance) - error handling
# ---------------------------------------------------------------------------

class TestFetchPriceHistory:
    def test_returns_dataframe_on_success(self, monkeypatch):
        df = make_ohlcv([100.0] * 120)
        patch_ticker(monkeypatch, history_result=df)
        result = tg.fetch_price_history("NVDA")
        assert len(result) == 120

    def test_requests_six_months_of_daily_data(self, monkeypatch):
        captured = {}

        class SpyTicker(FakeTicker):
            def history(self, *args, **kwargs):
                captured.update(kwargs)
                return make_ohlcv([100.0] * 120)

        monkeypatch.setattr(tg.yf, "Ticker", lambda symbol: SpyTicker())
        tg.fetch_price_history("NVDA")
        assert captured.get("period") == "6mo"
        assert captured.get("interval") == "1d"

    def test_empty_data_raises_ticker_not_found(self, monkeypatch):
        patch_ticker(monkeypatch, history_result=pd.DataFrame())
        with pytest.raises(tg.DataFetchError) as exc:
            tg.fetch_price_history("ZZZZZZ")
        assert "ZZZZZZ" in str(exc.value)
        assert "לא נמצאו נתונים" in str(exc.value)

    def test_none_result_raises_ticker_not_found(self, monkeypatch):
        patch_ticker(monkeypatch, history_result=None)
        with pytest.raises(tg.DataFetchError):
            tg.fetch_price_history("ZZZZZZ")

    def test_all_nan_close_raises_ticker_not_found(self, monkeypatch):
        df = make_ohlcv([np.nan] * 10)
        patch_ticker(monkeypatch, history_result=df)
        with pytest.raises(tg.DataFetchError) as exc:
            tg.fetch_price_history("ZZZZZZ")
        assert "לא נמצאו נתונים" in str(exc.value)

    def test_network_error_raises_hebrew_network_message(self, monkeypatch):
        patch_ticker(monkeypatch, history_exc=ConnectionError("boom"))
        with pytest.raises(tg.DataFetchError) as exc:
            tg.fetch_price_history("NVDA")
        assert "רשת" in str(exc.value)

    def test_unexpected_api_error_is_wrapped(self, monkeypatch):
        patch_ticker(monkeypatch, history_exc=ValueError("weird payload"))
        with pytest.raises(tg.DataFetchError) as exc:
            tg.fetch_price_history("NVDA")
        assert "שגיאה" in str(exc.value)

    def test_rate_limit_error_has_specific_message(self, monkeypatch):
        patch_ticker(monkeypatch, history_exc=tg.YFRateLimitError())
        with pytest.raises(tg.DataFetchError) as exc:
            tg.fetch_price_history("NVDA")
        assert "בקשות" in str(exc.value)

    def test_yfinance_ticker_missing_error_maps_to_not_found(self, monkeypatch):
        exc = tg.YFTickerMissingError("ZZZZZZ", "no data")
        patch_ticker(monkeypatch, history_exc=exc)
        with pytest.raises(tg.DataFetchError) as err:
            tg.fetch_price_history("ZZZZZZ")
        assert "לא נמצאו נתונים" in str(err.value)

    def test_yfinance_prices_missing_error_maps_to_not_found(self, monkeypatch):
        exc = tg.YFPricesMissingError("ZZZZZZ", "")
        patch_ticker(monkeypatch, history_exc=exc)
        with pytest.raises(tg.DataFetchError) as err:
            tg.fetch_price_history("ZZZZZZ")
        assert "לא נמצאו נתונים" in str(err.value)

    class FakeHTTPError(OSError):
        """Mimic curl_cffi's HTTPError: an OSError carrying a response."""

        def __init__(self, status):
            super().__init__(f"HTTP Error {status}")
            self.response = type("Resp", (), {"status_code": status})()

    @classmethod
    def http_error(cls, status):
        return cls.FakeHTTPError(status)

    def test_http_404_maps_to_not_found(self, monkeypatch):
        # Yahoo answers an unknown symbol with HTTP 404.
        patch_ticker(monkeypatch, history_exc=self.http_error(404))
        with pytest.raises(tg.DataFetchError) as err:
            tg.fetch_price_history("ZZZZZZ")
        assert "לא נמצאו נתונים" in str(err.value)

    def test_status_zero_response_maps_to_network_error(self, monkeypatch):
        # curl_cffi attaches a dummy Response(status_code=0) when the
        # connection never reached the server.
        patch_ticker(monkeypatch, history_exc=self.http_error(0))
        with pytest.raises(tg.DataFetchError) as err:
            tg.fetch_price_history("NVDA")
        assert "רשת" in str(err.value)

    def test_http_429_maps_to_rate_limit(self, monkeypatch):
        patch_ticker(monkeypatch, history_exc=self.http_error(429))
        with pytest.raises(tg.DataFetchError) as err:
            tg.fetch_price_history("NVDA")
        assert "בקשות" in str(err.value)

    def test_http_5xx_maps_to_service_error_not_network(self, monkeypatch):
        patch_ticker(monkeypatch, history_exc=self.http_error(503))
        with pytest.raises(tg.DataFetchError) as err:
            tg.fetch_price_history("NVDA")
        assert "אינו זמין" in str(err.value)
        assert "חיבור האינטרנט" not in str(err.value)

    def test_enables_yfinance_exceptions_so_network_errors_surface(self, monkeypatch):
        # By default yfinance swallows network errors and returns an empty
        # frame, which would be misreported as "ticker not found".
        monkeypatch.setattr(tg.yf.config.debug, "hide_exceptions", True)
        patch_ticker(monkeypatch, history_result=make_ohlcv([100.0] * 60))
        tg.fetch_price_history("NVDA")
        assert tg.yf.config.debug.hide_exceptions is False

    def test_missing_required_columns_raises(self, monkeypatch):
        df = make_ohlcv([100.0] * 60).drop(columns=["Volume"])
        patch_ticker(monkeypatch, history_result=df)
        with pytest.raises(tg.DataFetchError):
            tg.fetch_price_history("NVDA")


# ---------------------------------------------------------------------------
# Incomplete (still-trading) session handling
# ---------------------------------------------------------------------------

def make_tz_ohlcv(last_day, periods=60, tz="America/New_York"):
    index = pd.bdate_range(end=last_day, periods=periods, tz=tz)
    df = make_ohlcv([100.0] * periods)
    df.index = index
    return df


class TestDropIncompleteSession:
    def test_drops_todays_bar_while_market_is_open(self):
        df = make_tz_ohlcv("2026-10-09")
        now = pd.Timestamp("2026-10-09 12:00", tz="America/New_York")
        out, dropped = tg.drop_incomplete_session(df, now=now)
        assert dropped is True
        assert len(out) == len(df) - 1
        assert out.index[-1].date() == pd.Timestamp("2026-10-08").date()

    def test_keeps_todays_bar_after_market_close(self):
        df = make_tz_ohlcv("2026-10-09")
        now = pd.Timestamp("2026-10-09 17:30", tz="America/New_York")
        out, dropped = tg.drop_incomplete_session(df, now=now)
        assert dropped is False
        assert len(out) == len(df)

    def test_keeps_previous_days_bar(self):
        df = make_tz_ohlcv("2026-10-08")
        now = pd.Timestamp("2026-10-09 12:00", tz="America/New_York")
        out, dropped = tg.drop_incomplete_session(df, now=now)
        assert dropped is False
        assert len(out) == len(df)

    def test_uses_exchange_timezone_not_local_clock(self):
        # 19:00 in Israel == 12:00 in New York -> NY session still open.
        df = make_tz_ohlcv("2026-10-09")
        now = pd.Timestamp("2026-10-09 19:00", tz="Asia/Jerusalem")
        _, dropped = tg.drop_incomplete_session(df, now=now)
        assert dropped is True

    def test_naive_index_is_supported(self):
        df = make_ohlcv([100.0] * 60)  # naive index ending 2026-10-08
        now = pd.Timestamp("2026-10-08 10:00")
        _, dropped = tg.drop_incomplete_session(df, now=now)
        assert dropped is True


# ---------------------------------------------------------------------------
# Indicators (pandas-ta)
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def test_adds_expected_columns(self):
        df = make_ohlcv(np.linspace(100, 150, 120))
        out = tg.compute_indicators(df)
        for col in ("SMA_50", "RSI_14", "Volume_SMA_20"):
            assert col in out.columns

    def test_sma50_matches_mean_of_last_50_closes(self):
        closes = np.linspace(100, 150, 120)
        out = tg.compute_indicators(make_ohlcv(closes))
        assert out["SMA_50"].iloc[-1] == pytest.approx(closes[-50:].mean())

    def test_volume_sma20_matches_mean_of_last_20_volumes(self):
        vols = np.arange(1, 121) * 1000
        out = tg.compute_indicators(make_ohlcv([100.0] * 120, vols))
        assert out["Volume_SMA_20"].iloc[-1] == pytest.approx(vols[-20:].mean())

    def test_rsi_is_within_bounds(self):
        rng = np.random.default_rng(0)
        closes = 100 + rng.normal(0, 1, 120).cumsum()
        out = tg.compute_indicators(make_ohlcv(closes))
        assert 0 <= out["RSI_14"].iloc[-1] <= 100

    def test_rsi_high_on_steady_uptrend(self):
        out = tg.compute_indicators(make_ohlcv(np.linspace(100, 200, 120)))
        assert out["RSI_14"].iloc[-1] > 70

    def test_adds_atr14_column(self):
        out = tg.compute_indicators(make_ohlcv(np.linspace(100, 150, 120)))
        assert "ATR_14" in out.columns

    def test_atr_equals_constant_true_range(self):
        # Flat close at 100 with High=101, Low=99 -> True Range is 2 every day.
        out = tg.compute_indicators(make_ohlcv([100.0] * 120))
        assert out["ATR_14"].iloc[-1] == pytest.approx(2.0)

    def test_does_not_mutate_input(self):
        df = make_ohlcv(np.linspace(100, 150, 120))
        tg.compute_indicators(df)
        assert "SMA_50" not in df.columns


class TestLatestSnapshot:
    def test_daily_change_vs_previous_close(self):
        closes = np.linspace(100, 150, 120)
        s = tg.latest_snapshot(tg.compute_indicators(make_ohlcv(closes)))
        assert s.prev_close == pytest.approx(closes[-2])
        assert s.daily_change == pytest.approx(closes[-1] - closes[-2])
        assert s.daily_change_pct == pytest.approx((closes[-1] / closes[-2] - 1) * 100)

    def test_returns_last_row_values(self):
        closes = np.linspace(100, 150, 120)
        out = tg.compute_indicators(make_ohlcv(closes))
        snap = tg.latest_snapshot(out)
        assert snap.close == pytest.approx(150.0)
        assert snap.sma_50 == pytest.approx(closes[-50:].mean())
        assert snap.date == out.index[-1].date()
        assert snap.atr_14 == pytest.approx(out["ATR_14"].iloc[-1])

    def test_insufficient_history_raises_hebrew_error(self):
        # 30 trading days are not enough for a 50-day SMA.
        out = tg.compute_indicators(make_ohlcv(np.linspace(100, 110, 30)))
        with pytest.raises(tg.DataFetchError) as exc:
            tg.latest_snapshot(out)
        assert "50" in str(exc.value)


# ---------------------------------------------------------------------------
# 3-level classification
# ---------------------------------------------------------------------------

def snap(
    close=110.0, sma_50=100.0, rsi=40.0, volume=900.0, volume_sma_20=1000.0, atr=2.0,
    prev_close=None,
):
    return tg.Snapshot(
        date=date(2026, 10, 8),
        close=close,
        sma_50=sma_50,
        rsi_14=rsi,
        volume=volume,
        volume_sma_20=volume_sma_20,
        atr_14=atr,
        prev_close=prev_close,
    )


class TestDailyChange:
    def test_gain(self):
        s = snap(close=110.0, prev_close=100.0)
        assert s.daily_change == pytest.approx(10.0)
        assert s.daily_change_pct == pytest.approx(10.0)

    def test_loss(self):
        s = snap(close=95.0, prev_close=100.0)
        assert s.daily_change == pytest.approx(-5.0)
        assert s.daily_change_pct == pytest.approx(-5.0)

    def test_unknown_previous_close(self):
        s = snap(prev_close=None)
        assert s.daily_change is None
        assert s.daily_change_pct is None

    def test_zero_previous_close_has_no_percent(self):
        assert snap(close=1.0, prev_close=0.0).daily_change_pct is None

    def test_daily_change_does_not_affect_classification(self):
        base = snap(close=110.0, rsi=40.0)
        assert tg.classify(base) == tg.classify(snap(close=110.0, rsi=40.0, prev_close=200.0))


class TestClassify:
    def test_green_when_above_sma_low_rsi_low_volume(self):
        assert tg.classify(snap()) == tg.Level.GREEN

    def test_yellow_when_rsi_at_or_above_45(self):
        assert tg.classify(snap(rsi=45.0)) == tg.Level.YELLOW
        assert tg.classify(snap(rsi=60.0)) == tg.Level.YELLOW

    def test_yellow_when_volume_at_or_above_average(self):
        assert tg.classify(snap(volume=1000.0)) == tg.Level.YELLOW
        assert tg.classify(snap(volume=2500.0)) == tg.Level.YELLOW

    def test_red_when_below_sma50(self):
        assert tg.classify(snap(close=90.0)) == tg.Level.RED

    def test_red_overrides_good_rsi_and_volume(self):
        assert tg.classify(snap(close=90.0, rsi=30.0, volume=10.0)) == tg.Level.RED

    def test_close_equal_to_sma50_is_red(self):
        # Not *above* the trend line -> conservative: disqualified.
        assert tg.classify(snap(close=100.0)) == tg.Level.RED


# ---------------------------------------------------------------------------
# Capital shield / risk management (PRD 3.E)
# ---------------------------------------------------------------------------

class TestRiskConstants:
    def test_default_capital_and_risk(self):
        assert tg.VIRTUAL_CAPITAL == 10_000
        assert tg.RISK_PER_TRADE == 0.01


class TestComputeRiskPlan:
    def test_basic_plan(self):
        plan = tg.compute_risk_plan(snap(close=100.0, atr=2.0))
        assert plan.entry == pytest.approx(100.0)
        assert plan.atr == pytest.approx(2.0)
        assert plan.stop_loss == pytest.approx(96.0)       # 100 - 2*2
        assert plan.risk_per_share == pytest.approx(4.0)
        assert plan.take_profit == pytest.approx(108.0)    # 100 + 2*4 (1:2)
        assert plan.risk_budget == pytest.approx(100.0)    # 1% of 10,000
        assert plan.shares == 25                           # 100 / 4
        assert plan.position_value == pytest.approx(2500.0)
        assert plan.max_loss == pytest.approx(100.0)
        assert plan.expected_profit == pytest.approx(200.0)  # 25 * (108 - 100)
        assert plan.capped_by_capital is False
        assert plan.is_valid is True

    def test_shares_are_floored(self):
        plan = tg.compute_risk_plan(snap(close=100.0, atr=3.0))
        assert plan.stop_loss == pytest.approx(94.0)
        assert plan.take_profit == pytest.approx(112.0)
        assert plan.shares == 16  # 100 / 6 = 16.67 -> 16
        assert plan.max_loss == pytest.approx(96.0)
        assert plan.expected_profit == pytest.approx(192.0)  # 16 * (112 - 100)

    def test_custom_capital_and_risk(self):
        plan = tg.compute_risk_plan(
            snap(close=100.0, atr=2.0), capital=50_000, risk_pct=0.02
        )
        assert plan.risk_budget == pytest.approx(1000.0)
        assert plan.shares == 250

    def test_shares_capped_by_available_capital(self):
        # Tiny ATR -> risk-based size (500 shares = $50,000) exceeds capital.
        plan = tg.compute_risk_plan(snap(close=100.0, atr=0.1))
        assert plan.shares == 100  # 10,000 / 100
        assert plan.capped_by_capital is True
        assert plan.position_value <= tg.VIRTUAL_CAPITAL

    def test_expected_profit_is_twice_max_loss_at_1_to_2(self):
        plan = tg.compute_risk_plan(snap(close=100.0, atr=3.0))
        assert plan.expected_profit == pytest.approx(2 * plan.max_loss)

    def test_expected_profit_zero_when_no_shares(self):
        assert tg.compute_risk_plan(snap(close=1000.0, atr=60.0)).expected_profit == 0
        assert tg.compute_risk_plan(snap(close=100.0, atr=0.0)).expected_profit == 0

    def test_risk_per_share_above_budget_gives_zero_shares(self):
        plan = tg.compute_risk_plan(snap(close=1000.0, atr=60.0))  # risk 120 > 100
        assert plan.shares == 0
        assert plan.is_valid is True

    def test_zero_atr_is_invalid(self):
        plan = tg.compute_risk_plan(snap(close=100.0, atr=0.0))
        assert plan.is_valid is False
        assert plan.shares == 0

    def test_non_positive_stop_is_invalid(self):
        plan = tg.compute_risk_plan(snap(close=1.0, atr=1.0))  # stop = -1
        assert plan.is_valid is False
        assert plan.shares == 0


# ---------------------------------------------------------------------------
# TradingView link
# ---------------------------------------------------------------------------

class TestTradingViewUrl:
    def test_simple_ticker(self):
        assert tg.tradingview_url("NVDA") == "https://www.tradingview.com/chart/?symbol=NVDA"

    def test_dash_converted_to_dot(self):
        assert tg.tradingview_url("BRK-B").endswith("symbol=BRK.B")


# ---------------------------------------------------------------------------
# Report rendering (Hebrew)
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_green_report_contents(self):
        text = tg.build_report("NVDA", snap(), tg.Level.GREEN)
        assert "NVDA" in text
        assert "מתאימה להמשך בדיקה" in text
        assert "נר היפוך" in text
        assert "https://www.tradingview.com/chart/?symbol=NVDA" in text

    def test_yellow_report_contents(self):
        text = tg.build_report("NVDA", snap(rsi=60.0), tg.Level.YELLOW)
        assert "דורשת זהירות" in text
        assert "להמתין" in text

    def test_yellow_report_explains_which_condition_failed(self):
        rsi_text = tg.build_report("NVDA", snap(rsi=60.0), tg.Level.YELLOW)
        vol_text = tg.build_report("NVDA", snap(volume=5000.0), tg.Level.YELLOW)
        assert "RSI" in rsi_text and "45" in rsi_text
        assert "מחזור" in vol_text

    def test_red_report_contents(self):
        text = tg.build_report("NVDA", snap(close=90.0), tg.Level.RED)
        assert "נפסלת" in text
        assert "על הגדר" in text

    def test_report_shows_indicator_values(self):
        text = tg.build_report("NVDA", snap(), tg.Level.GREEN)
        assert "110.00" in text  # close
        assert "100.00" in text  # SMA_50
        assert "40.0" in text    # RSI

    def test_report_never_contains_risk_block_or_disclaimer(self):
        # Those are printed by main() only after the news veto step.
        text = tg.build_report("NVDA", snap(), tg.Level.GREEN)
        assert RISK_TITLE not in text
        assert DISCLAIMER not in text

    def test_tradingview_setup_instructions(self):
        text = tg.build_report("NVDA", snap(), tg.Level.RED)
        assert "הוראות הגדרה ב-TradingView:" in text
        assert "לחץ למעלה על כפתור 'Indicators' (סמל של fx)." in text
        assert "חפש 'SMA' (Moving Average Simple), לחץ עליו" in text
        assert "ושנה את ה-Length ל-50." in text
        assert "חפש 'RSI' (Relative Strength Index) ולחץ עליו." in text
        assert "ואת ה-RSI והמחזורים (Volume) בתחתית המסך." in text
        assert "טיפ:" not in text


RISK_TITLE = "מגן הון (ניהול סיכונים לעסקה מדומה)"
GAP_WARNING = (
    "שים לב: זהו תכנון מתמטי לימודי. במקרה של פער מחירים (Gap) בפתיחת המסחר, "
    "ההפסד בפועל עלול להיות גדול מ-1%"
)
DISCLAIMER = "אינה מספקת איתותי מסחר"


class TestBuildRiskBlock:
    def test_contents(self):
        text = tg.build_risk_block(snap(close=100.0, atr=2.0))
        assert RISK_TITLE in text
        assert "ATR" in text and "2.00" in text
        assert "96.00" in text   # stop-loss
        assert "108.00" in text  # take-profit
        assert "25" in text      # shares
        assert GAP_WARNING in text

    def test_uses_beginner_friendly_labels(self):
        text = tg.build_risk_block(snap(close=100.0, atr=2.0))
        assert "תנודתיות יומית ממוצעת (ATR):" in text
        assert "קו הגנה (Stop-Loss) - מרווח של פי 2 מהתנודתיות:" in text
        assert "הפסד אפשרי על כל מניה בודדת:" in text
        assert "ATR נוכחי" not in text
        assert "סיכון למניה" not in text

    def test_notes_capital_cap(self):
        assert "הוגבלה" in tg.build_risk_block(snap(close=100.0, atr=0.1))

    def test_explains_zero_shares(self):
        assert "אפילו מניה אחת" in tg.build_risk_block(snap(close=1000.0, atr=60.0))

    def test_expected_profit_is_the_last_row(self):
        text = tg.build_risk_block(snap(close=100.0, atr=2.0))
        rows = [line for line in text.splitlines() if ":" in line and "$" in line]
        assert "רווח מתוכנן אם היעד יושג:" in rows[-1]
        assert rows[-1].endswith("$200.00")

    def test_handles_invalid_plan(self):
        text = tg.build_risk_block(snap(close=100.0, atr=0.0))
        assert RISK_TITLE in text
        assert "לא ניתן לחשב" in text


class TestFooter:
    def test_footer_has_not_a_signal_disclaimer(self):
        assert DISCLAIMER in tg.build_footer()


# ---------------------------------------------------------------------------
# Earnings blackout (PRD 3.D)
# ---------------------------------------------------------------------------

class TestFetchNextEarningsDate:
    TODAY = date(2026, 10, 9)

    def fetch(self, monkeypatch, **kwargs):
        patch_ticker(monkeypatch, **kwargs)
        return tg.fetch_next_earnings_date("NVDA", today=self.TODAY)

    def test_reads_date_from_calendar(self, monkeypatch):
        cal = {"Earnings Date": [date(2026, 11, 17)]}
        assert self.fetch(monkeypatch, calendar=cal) == date(2026, 11, 17)

    def test_picks_earliest_upcoming_of_several(self, monkeypatch):
        cal = {"Earnings Date": [date(2026, 11, 20), date(2026, 11, 17)]}
        assert self.fetch(monkeypatch, calendar=cal) == date(2026, 11, 17)

    def test_ignores_past_dates(self, monkeypatch):
        cal = {"Earnings Date": [date(2026, 10, 1), date(2026, 10, 20)]}
        assert self.fetch(monkeypatch, calendar=cal) == date(2026, 10, 20)

    def test_today_counts_as_upcoming(self, monkeypatch):
        cal = {"Earnings Date": [self.TODAY]}
        assert self.fetch(monkeypatch, calendar=cal) == self.TODAY

    def test_only_past_dates_returns_none(self, monkeypatch):
        cal = {"Earnings Date": [date(2026, 10, 1)]}
        assert self.fetch(monkeypatch, calendar=cal) is None

    def test_accepts_datetime_timestamp_and_single_value(self, monkeypatch):
        cal = {"Earnings Date": [datetime(2026, 11, 17, 21, 0), pd.Timestamp("2026-11-18")]}
        assert self.fetch(monkeypatch, calendar=cal) == date(2026, 11, 17)
        cal = {"Earnings Date": pd.Timestamp("2026-11-18")}
        assert self.fetch(monkeypatch, calendar=cal) == date(2026, 11, 18)

    @pytest.mark.parametrize(
        "cal",
        [None, {}, {"Earnings Date": None}, {"Earnings Date": []},
         {"Earnings Date": ["not a date"]}, "garbage", pd.DataFrame()],
    )
    def test_missing_or_malformed_returns_none(self, monkeypatch, cal):
        assert self.fetch(monkeypatch, calendar=cal) is None

    def test_api_error_returns_none(self, monkeypatch):
        assert self.fetch(monkeypatch, calendar_exc=ConnectionError("x")) is None

    def test_ticker_construction_error_returns_none(self, monkeypatch):
        def boom(symbol):
            raise RuntimeError("yfinance exploded")

        monkeypatch.setattr(tg.yf, "Ticker", boom)
        assert tg.fetch_next_earnings_date("NVDA", today=self.TODAY) is None


class TestIsEarningsTooClose:
    TODAY = date(2026, 10, 9)

    @pytest.mark.parametrize("days, expected", [
        (0, True), (1, True), (14, True), (15, False), (40, False),
    ])
    def test_window(self, days, expected):
        earnings = self.TODAY + timedelta(days=days)
        assert tg.is_earnings_too_close(earnings, self.TODAY) is expected


class TestBuildEarningsBlock:
    def test_contents(self):
        text = tg.build_earnings_block("NVDA", date(2026, 10, 14), date(2026, 10, 9))
        assert "[אדום]" in text
        assert "נפסלת" in text
        assert "2026-10-14" in text
        assert "בעוד 5 ימים" in text
        assert "דוח כספי" in text

    def test_today_wording(self):
        text = tg.build_earnings_block("NVDA", date(2026, 10, 9), date(2026, 10, 9))
        assert "היום" in text


# ---------------------------------------------------------------------------
# News (PRD 3.D - Human-in-the-Loop)
# ---------------------------------------------------------------------------

class TestFetchNews:
    def fetch(self, monkeypatch, payload=None, exc=None, ticker="NVDA"):
        self.calls = patch_rss(monkeypatch, payload=payload, exc=exc)
        return tg.fetch_news(ticker)

    def test_parses_title_link_and_date(self, monkeypatch):
        xml = rss_xml(rss_item("Chip deal", "https://x.io/1"))
        items = self.fetch(monkeypatch, xml)
        assert items == [
            tg.NewsItem("Chip deal", "https://x.io/1", None, "2026-10-09")
        ]

    def test_ignores_channel_level_title_and_link(self, monkeypatch):
        # <channel>/<image> also carry <title>/<link>; only <item>s are news.
        items = self.fetch(monkeypatch, rss_xml(rss_item("Only item")))
        assert [i.title for i in items] == ["Only item"]

    def test_limits_to_five(self, monkeypatch):
        xml = rss_xml(*(rss_item(f"H{i}") for i in range(10)))
        items = self.fetch(monkeypatch, xml)
        assert [i.title for i in items] == ["H0", "H1", "H2", "H3", "H4"]
        assert tg.NEWS_LIMIT == 5

    def test_skips_items_without_title(self, monkeypatch):
        xml = rss_xml(rss_item(None), rss_item("   "), rss_item("Good"))
        assert [i.title for i in self.fetch(monkeypatch, xml)] == ["Good"]

    def test_missing_link_and_bad_date_are_tolerated(self, monkeypatch):
        xml = rss_xml(rss_item("No link", link=None, pub="not a date"))
        assert self.fetch(monkeypatch, xml) == [
            tg.NewsItem("No link", None, None, None)
        ]

    def test_decodes_entities_and_hebrew(self, monkeypatch):
        xml = rss_xml(rss_item("AT&amp;T &amp; שלום", "https://x.io/?a=1&amp;b=2"))
        item = self.fetch(monkeypatch, xml)[0]
        assert item.title == "AT&T & שלום"
        assert item.url == "https://x.io/?a=1&b=2"

    def test_empty_channel_returns_empty_list(self, monkeypatch):
        # Yahoo answers an unknown ticker with HTTP 200 and an empty channel.
        assert self.fetch(monkeypatch, rss_xml()) == []

    @pytest.mark.parametrize("payload", [
        b"", b"not xml at all", b"<rss><channel><item>", b"<html><body>Blocked</body></html>",
    ])
    def test_malformed_or_non_rss_returns_empty_list(self, monkeypatch, payload):
        assert self.fetch(monkeypatch, payload) == []

    @pytest.mark.parametrize("exc", [
        urllib.error.URLError("dns failure"),
        urllib.error.HTTPError("u", 503, "Service Unavailable", Message(), None),
        TimeoutError("timed out"),
        OSError("connection reset"),
    ])
    def test_network_errors_return_empty_list(self, monkeypatch, exc):
        assert self.fetch(monkeypatch, exc=exc) == []

    def test_requests_yahoo_rss_url_for_ticker(self, monkeypatch):
        self.fetch(monkeypatch, rss_xml())
        assert self.calls == [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US"
        ]

    def test_ticker_is_url_encoded(self, monkeypatch):
        self.fetch(monkeypatch, rss_xml(), ticker="^GSPC")
        assert "s=%5EGSPC&" in self.calls[0]


class FakeResponse:
    """Minimal context-manager stand-in for urlopen()'s response."""

    def __init__(self, body):
        self.body = body
        self.read_limit = None

    def read(self, amt=None):
        self.read_limit = amt
        return self.body if amt is None else self.body[:amt]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDownloadRss:
    def run(self, monkeypatch, body=b"<rss/>"):
        captured = {}
        response = FakeResponse(body)

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        monkeypatch.setattr(tg.urllib.request, "urlopen", fake_urlopen)
        # Undo the autouse stub so the real _download_rss runs.
        monkeypatch.setattr(tg, "_download_rss", ORIGINAL_DOWNLOAD_RSS)
        result = tg._download_rss("https://feeds.example/rss")
        return result, captured, response

    def test_sends_browser_user_agent(self, monkeypatch):
        _, captured, _ = self.run(monkeypatch)
        assert captured["request"].get_header("User-agent").startswith("Mozilla/5.0")

    def test_uses_five_second_timeout(self, monkeypatch):
        _, captured, _ = self.run(monkeypatch)
        assert captured["timeout"] == 5

    def test_returns_body_bytes(self, monkeypatch):
        body, _, _ = self.run(monkeypatch, b"<rss>x</rss>")
        assert body == b"<rss>x</rss>"

    def test_caps_download_size(self, monkeypatch):
        _, _, response = self.run(monkeypatch)
        assert response.read_limit == tg.RSS_MAX_BYTES


class TestBuildNewsBlock:
    def test_lists_titles_and_links(self):
        items = [
            tg.NewsItem("First headline", "https://a.io", "Reuters", "2026-10-08"),
            tg.NewsItem("Second headline", None, None, None),
        ]
        text = tg.build_news_block("NVDA", items)
        assert "חדשות ואזהרות" in text
        assert "1. First headline" in text
        assert "https://a.io" in text
        assert "Reuters" in text
        assert "2. Second headline" in text

    def test_empty_news_points_to_manual_check(self):
        text = tg.build_news_block("NVDA", [])
        assert "[צהוב] לא נמצאו כותרות" in text
        assert "https://finance.yahoo.com/quote/NVDA/news" in text


class TestConfigureConsole:
    def test_piped_stdin_is_decoded_as_utf8(self, monkeypatch):
        # Windows decodes piped stdin with the ANSI code page by default,
        # which garbles a Hebrew veto answer such as 'לא'.
        piped = io.TextIOWrapper(io.BytesIO("לא\n".encode("utf-8")), encoding="cp1252")
        monkeypatch.setattr(sys, "stdin", piped)
        tg._configure_console()
        assert sys.stdin.readline().strip() == "לא"


VETO_PROMPT = (
    "האם זיהית בחדשות אירוע חריג שדורש פסילה? "
    "(הקלד 'כן' כדי לפסול, או 'לא' כדי להמשיך לניהול הסיכונים): "
)


def feed_input(monkeypatch, answers):
    """Patch input() to return `answers` in order; record prompts asked.

    Raises EOFError when answers run out - like a closed stdin - so a test
    can never hang on an unexpected extra prompt.
    """
    it = iter(answers)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


class TestAskNewsVeto:
    @pytest.mark.parametrize("answer", ["כן", " כן ", "yes", "Y"])
    def test_yes_disqualifies(self, monkeypatch, answer):
        feed_input(monkeypatch, [answer])
        assert tg.ask_news_veto() is True

    @pytest.mark.parametrize("answer", ["לא", " לא ", "no", "N"])
    def test_no_continues(self, monkeypatch, answer):
        feed_input(monkeypatch, [answer])
        assert tg.ask_news_veto() is False

    def test_uses_exact_hebrew_prompt(self, monkeypatch):
        prompts = feed_input(monkeypatch, ["לא"])
        tg.ask_news_veto()
        assert prompts == [VETO_PROMPT]

    def test_reasks_on_unclear_answer(self, monkeypatch, capsys):
        prompts = feed_input(monkeypatch, ["אולי", "", "לא"])
        assert tg.ask_news_veto() is False
        assert len(prompts) == 3
        assert "'כן' או 'לא'" in capsys.readouterr().out

    def test_disqualifies_after_too_many_unclear_answers(self, monkeypatch, capsys):
        prompts = feed_input(monkeypatch, ["?", "?", "?", "לא"])
        assert tg.ask_news_veto() is True
        assert len(prompts) == tg.MAX_VETO_ATTEMPTS == 3
        assert "ליתר ביטחון" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End-to-end CLI flow (main)
# ---------------------------------------------------------------------------

UPTREND = np.linspace(100, 200, 120)    # above SMA_50 -> YELLOW (RSI high)
DOWNTREND = np.linspace(200, 100, 120)  # below SMA_50 -> RED
EARNINGS_UNKNOWN = "לא אותר תאריך דוח כספי. באחריותך לבדוק זאת ידנית לפני כניסה לעסקה."


class TestMain:
    def run_main(self, monkeypatch, capsys, answers, rss=None, **ticker_kwargs):
        if isinstance(answers, str):
            answers = [answers]
        self.prompts = feed_input(monkeypatch, answers)
        if rss is not None:
            self.rss_calls = patch_rss(monkeypatch, payload=rss)
        patch_ticker(monkeypatch, **ticker_kwargs)
        code = tg.main()
        out, err = capsys.readouterr()
        return code, out + err

    def test_full_flow_with_no_veto_prints_risk_block(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["nvda", "לא"],
            history_result=make_ohlcv(UPTREND),
            rss=rss_xml(rss_item("Big headline")),
        )
        assert code == 0
        assert "NVDA" in output
        assert "Big headline" in output
        # Order: report -> news -> risk block -> disclaimer at the very end.
        assert (
            output.index("tradingview.com")
            < output.index("Big headline")
            < output.index(RISK_TITLE)
            < output.index(DISCLAIMER)
        )
        assert self.prompts[1] == VETO_PROMPT

    def test_veto_yes_disqualifies_without_risk_block(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA", "כן"], history_result=make_ohlcv(UPTREND)
        )
        assert code == 0
        assert "[אדום] נפסלת" in output
        assert RISK_TITLE not in output
        assert DISCLAIMER in output

    def test_earnings_within_14_days_stops_immediately(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA"],
            history_result=make_ohlcv(UPTREND),
            calendar={"Earnings Date": [days_from_today(5)]},
            rss=rss_xml(rss_item("Should not be shown")),
        )
        assert code == 0
        assert "[אדום]" in output and "דוח כספי" in output
        assert "דוח לימודי טכני" not in output  # analysis stopped
        assert "Should not be shown" not in output
        assert self.rss_calls == []  # news never fetched
        assert RISK_TITLE not in output
        assert len(self.prompts) == 1  # never asked the veto question

    def test_earnings_today_stops_immediately(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA"],
            history_result=make_ohlcv(UPTREND),
            calendar={"Earnings Date": [date.today()]},
        )
        assert code == 0
        assert "דוח לימודי טכני" not in output

    def test_far_earnings_date_is_shown_and_flow_continues(self, monkeypatch, capsys):
        far = days_from_today(40)
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA", "לא"],
            history_result=make_ohlcv(UPTREND),
            calendar={"Earnings Date": [far]},
        )
        assert code == 0
        assert far.isoformat() in output
        assert EARNINGS_UNKNOWN not in output
        assert RISK_TITLE in output

    def test_unknown_earnings_warns_yellow_and_continues(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA", "לא"],
            history_result=make_ohlcv(UPTREND), calendar=None,
        )
        assert code == 0
        assert f"[צהוב] {EARNINGS_UNKNOWN}" in output
        assert RISK_TITLE in output

    def test_earnings_and_news_api_errors_do_not_crash(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA", "לא"],
            history_result=make_ohlcv(UPTREND),
            calendar_exc=RuntimeError("Yahoo flaked"),
        )  # the autouse fixture makes the RSS download fail
        assert code == 0
        assert EARNINGS_UNKNOWN in output
        assert "לא נמצאו כותרות" in output
        assert RISK_TITLE in output
        assert "Traceback" not in output

    def test_red_stock_skips_news_and_veto(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA"],
            history_result=make_ohlcv(DOWNTREND),
            rss=rss_xml(rss_item("Irrelevant")),
        )
        assert code == 0
        assert "[אדום] נפסלת" in output
        assert "Irrelevant" not in output
        assert self.rss_calls == []  # news never fetched
        assert RISK_TITLE not in output
        assert DISCLAIMER in output
        assert len(self.prompts) == 1

    def test_eof_at_veto_prompt_exits_gracefully(self, monkeypatch, capsys):
        # Only the ticker is supplied: the veto prompt hits EOF (Ctrl+Z/Ctrl+D).
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA"], history_result=make_ohlcv(UPTREND)
        )
        assert code == 1
        assert "בוטלה" in output
        assert RISK_TITLE not in output
        assert "Traceback" not in output

    def test_notes_when_todays_open_session_is_excluded(self, monkeypatch, capsys):
        df = make_ohlcv(UPTREND)  # last bar 2026-10-08
        monkeypatch.setattr(
            tg, "_now", lambda tz: pd.Timestamp("2026-10-08 11:00", tz=tz)
        )
        code, output = self.run_main(
            monkeypatch, capsys, ["NVDA", "לא"], history_result=df
        )
        assert code == 0
        assert "המסחר היום עדיין פתוח" in output
        assert "2026-10-07" in output

    def test_unknown_ticker_returns_one_without_traceback(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, "ZZZZZZ", history_result=pd.DataFrame()
        )
        assert code == 1
        assert "לא נמצאו נתונים" in output
        assert "Traceback" not in output

    def test_network_error_returns_one_without_traceback(self, monkeypatch, capsys):
        code, output = self.run_main(
            monkeypatch, capsys, "NVDA", history_exc=ConnectionError("down")
        )
        assert code == 1
        assert "רשת" in output
        assert "Traceback" not in output

    def test_empty_input_returns_one(self, monkeypatch, capsys):
        code, output = self.run_main(monkeypatch, capsys, "  ", history_result=None)
        assert code == 1
        assert "לא הוזן" in output

    def test_ctrl_c_exits_gracefully(self, monkeypatch, capsys):
        def raise_interrupt(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_interrupt)
        code = tg.main()
        out, err = capsys.readouterr()
        assert code == 1
        assert "Traceback" not in out + err

    def test_unexpected_internal_error_is_caught(self, monkeypatch, capsys):
        monkeypatch.setattr(tg, "compute_indicators", lambda d: 1 / 0)
        code, output = self.run_main(
            monkeypatch, capsys, "NVDA", history_result=make_ohlcv(UPTREND)
        )
        assert code == 1
        assert "שגיאה" in output
        assert "Traceback" not in output
