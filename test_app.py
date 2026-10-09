"""Tests for app.py (TradeGuardian Streamlit GUI).

Uses Streamlit's headless AppTest runner. The app runs in-process, so the
same monkeypatches as the CLI tests (yfinance.Ticker, RSS download) apply.
No test touches the network.
"""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import tg_mentor_mvp as tg
from test_tg_mentor_mvp import FakeTicker, days_from_today, make_ohlcv, rss_item, rss_xml

APP_PATH = str(Path(__file__).with_name("app.py"))

UPTREND = np.linspace(100, 200, 120)    # above SMA_50, RSI high -> YELLOW
DOWNTREND = np.linspace(200, 100, 120)  # below SMA_50 -> RED
# Uptrend then a calm 15-day pullback: above SMA_50, RSI ~41, low last volume.
GREEN_CLOSES = np.concatenate([np.linspace(100, 160, 105), np.linspace(160, 154, 15)])
GREEN_VOLUMES = np.r_[np.full(119, 1_000_000.0), 500_000.0]

RISK_TITLE = "מגן הון (ניהול סיכונים לעסקה מדומה)"
DISCLAIMER = "אינה מספקת איתותי מסחר"


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def offline_news(monkeypatch):
    """By default the RSS download fails (no network in tests)."""
    def fail(url):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(tg, "_download_rss", fail)


def patch_market(monkeypatch, **kwargs):
    monkeypatch.setattr(tg.yf, "Ticker", lambda symbol: FakeTicker(**kwargs))


def patch_news(monkeypatch, payload):
    calls = []

    def fake_download(url):
        calls.append(url)
        return payload

    monkeypatch.setattr(tg, "_download_rss", fake_download)
    return calls


def new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


def run_check(ticker: str = "NVDA") -> AppTest:
    at = new_app()
    at.text_input(key="ticker").input(ticker)
    at.button(key="check").click()
    at.run()
    assert not at.exception, at.exception
    return at


def text_of(elements) -> str:
    return "\n".join(str(e.value) for e in elements)


def page_text(at: AppTest) -> str:
    """All visible text: alerts, markdown, headers, captions and metrics."""
    parts = [
        text_of(at.title), text_of(at.subheader), text_of(at.markdown),
        text_of(at.caption), text_of(at.success), text_of(at.info),
        text_of(at.warning), text_of(at.error),
        "\n".join(f"{m.label} {m.value}" for m in at.metric),
    ]
    return "\n".join(parts)


def button_keys(at: AppTest) -> set[str]:
    return {b.key for b in at.button if b.key}


# ---------------------------------------------------------------------------
# Layout, RTL and input
# ---------------------------------------------------------------------------

class TestInitialScreen:
    def test_renders_input_and_button_without_errors(self):
        at = new_app()
        assert not at.exception
        assert at.text_input(key="ticker") is not None
        assert at.button(key="check").label == "בדוק מניה"
        assert not at.success and not at.warning and not at.error

    def test_injects_rtl_css(self):
        css = text_of(new_app().markdown)
        assert "direction: rtl" in css
        assert "text-align: right" in css

    def test_shows_disclaimer(self):
        assert DISCLAIMER in page_text(new_app())

    def test_rtl_is_scoped_to_content_not_the_page_frame(self):
        # On mobile Streamlit hides the collapsed sidebar with
        # translateX(-width), which assumes an LTR frame. Flipping the frame
        # (html/body/.stApp/stSidebar) to RTL drags the sidebar over the page.
        css = text_of(new_app().markdown)
        rtl_rule = css.split("{")[0]
        assert '[data-testid="stMain"]' in rtl_rule
        assert '[data-testid="stSidebarUserContent"]' in rtl_rule
        for frame in ("html", "body", ".stApp", '[data-testid="stSidebar"]',
                      '[data-testid="stAppViewContainer"]'):
            assert frame not in [s.strip() for s in rtl_rule.split(",")], frame

    def test_css_forces_right_alignment_on_headings_and_markdown(self):
        # Streamlit sets text-align:left on headings/markdown containers, which
        # overrides an inherited RTL alignment unless forced.
        css = text_of(new_app().markdown)
        assert '[data-testid="stHeading"]' in css
        assert '[data-testid="stMarkdown"]' in css
        assert "text-align: right !important" in css


# ---------------------------------------------------------------------------
# Sidebar: virtual portfolio settings
# ---------------------------------------------------------------------------

class TestSidebar:
    def test_header_and_inputs_live_in_sidebar(self):
        at = new_app()
        assert "הגדרות תיק וירטואלי" in text_of(at.sidebar.header)
        sidebar_keys = {w.key for w in at.sidebar.number_input}
        assert {"capital", "risk_pct"} <= sidebar_keys

    def test_capital_input_defaults(self):
        capital = new_app().number_input(key="capital")
        assert capital.label == "הון וירטואלי בדולרים"
        assert capital.value == 10_000
        assert capital.step == 1000

    def test_risk_input_defaults(self):
        risk = new_app().number_input(key="risk_pct")
        assert risk.label == "סיכון מקסימלי לעסקה (%)"
        assert risk.value == pytest.approx(1.0)
        assert risk.step == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# RTL-safe wording for mixed Hebrew/English
# ---------------------------------------------------------------------------

class TestMixedDirectionText:
    def test_news_heading_keeps_english_on_its_own_line(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND))
        at = run_check()
        assert "חדשות ואזהרות" in [str(h.value) for h in at.subheader]
        assert not any("Human-in-the-Loop" in str(h.value) for h in at.subheader)
        assert "Human-in-the-Loop" in text_of(at.caption)

    def test_tradingview_link_text_ends_with_english(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND))
        md = text_of(run_check().markdown)
        assert (
            "[לחץ כאן לפתיחת הגרף המלא ב-TradingView]"
            "(https://www.tradingview.com/chart/?symbol=NVDA)"
        ) in md
        assert "פתח את גרף המניה" not in md


class TestInputErrors:
    def test_empty_ticker(self):
        at = run_check("   ")
        assert "לא הוזן" in text_of(at.error)

    def test_invalid_ticker(self):
        at = run_check("NV DA!")
        assert "אינו תקין" in text_of(at.error)

    def test_unknown_ticker(self, monkeypatch):
        import pandas as pd

        patch_market(monkeypatch, history_result=pd.DataFrame())
        at = run_check("ZZZZZZ")
        assert "לא נמצאו נתונים" in text_of(at.error)

    def test_network_error(self, monkeypatch):
        patch_market(monkeypatch, history_exc=ConnectionError("down"))
        assert "רשת" in text_of(run_check().error)

    def test_unexpected_error_is_caught_without_traceback(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND))
        monkeypatch.setattr(tg, "compute_indicators", lambda df: 1 / 0)
        at = run_check()
        assert "שגיאה לא צפויה" in text_of(at.error)


# ---------------------------------------------------------------------------
# Technical report & 3 levels
# ---------------------------------------------------------------------------

class TestLevels:
    def test_green_uses_success_box(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(GREEN_CLOSES, GREEN_VOLUMES))
        at = run_check()
        assert "מתאימה להמשך בדיקה" in text_of(at.success)
        assert "נר היפוך" in text_of(at.success)

    def test_yellow_uses_warning_box(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND))
        at = run_check()
        assert "דורשת זהירות" in text_of(at.warning)
        assert "RSI" in text_of(at.warning)

    def test_red_uses_error_box_and_stops(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(DOWNTREND))
        calls = patch_news(monkeypatch, rss_xml(rss_item("Irrelevant")))
        at = run_check()
        assert "נפסלת" in text_of(at.error)
        assert "על הגדר" in text_of(at.error)
        assert {"approve", "veto"}.isdisjoint(button_keys(at))
        assert calls == []  # no news for a disqualified stock
        assert RISK_TITLE not in page_text(at)

    def test_report_shows_indicator_metrics(self, monkeypatch):
        df = make_ohlcv(GREEN_CLOSES, GREEN_VOLUMES)
        patch_market(monkeypatch, history_result=df)
        snap = tg.latest_snapshot(tg.compute_indicators(df))
        text = page_text(run_check())
        assert "NVDA" in text
        assert f"{snap.close:,.2f}" in text
        assert f"{snap.sma_50:,.2f}" in text
        assert f"{snap.rsi_14:.1f}" in text

    def test_report_links_to_tradingview(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND))
        at = run_check()
        assert "https://www.tradingview.com/chart/?symbol=NVDA" in page_text(at)
        assert "הוראות הגדרה ב-TradingView" in page_text(at)


# ---------------------------------------------------------------------------
# Earnings blackout
# ---------------------------------------------------------------------------

class TestEarnings:
    def test_earnings_within_14_days_blocks_everything(self, monkeypatch):
        patch_market(
            monkeypatch,
            history_result=make_ohlcv(UPTREND),
            calendar={"Earnings Date": [days_from_today(5)]},
        )
        calls = patch_news(monkeypatch, rss_xml(rss_item("Hidden")))
        at = run_check()
        assert "דוח כספי" in text_of(at.error)
        assert "בעוד 5 ימים" in text_of(at.error)
        assert "דוח לימודי טכני" not in page_text(at)
        assert {"approve", "veto"}.isdisjoint(button_keys(at))
        assert calls == []

    def test_unknown_earnings_shows_yellow_warning(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND), calendar=None)
        assert tg.EARNINGS_UNKNOWN_MESSAGE in text_of(run_check().warning)

    def test_far_earnings_date_is_shown_as_info(self, monkeypatch):
        far = days_from_today(40)
        patch_market(
            monkeypatch,
            history_result=make_ohlcv(UPTREND),
            calendar={"Earnings Date": [far]},
        )
        at = run_check()
        assert far.isoformat() in text_of(at.info)
        assert tg.EARNINGS_UNKNOWN_MESSAGE not in page_text(at)


# ---------------------------------------------------------------------------
# News & veto (Human-in-the-Loop) + capital shield
# ---------------------------------------------------------------------------

class TestNewsAndVeto:
    def setup_green(self, monkeypatch, news=None):
        self.df = make_ohlcv(GREEN_CLOSES, GREEN_VOLUMES)
        patch_market(monkeypatch, history_result=self.df)
        if news is not None:
            patch_news(monkeypatch, news)

    def test_headlines_and_links_are_listed(self, monkeypatch):
        self.setup_green(monkeypatch, rss_xml(
            rss_item("Big headline", "https://news.io/1"),
            rss_item("Second story", "https://news.io/2"),
        ))
        text = page_text(run_check())
        assert "Big headline" in text and "https://news.io/1" in text
        assert "Second story" in text

    def test_dollar_signs_in_headlines_are_not_rendered_as_math(self, monkeypatch):
        # Real headline: two '$' would make Streamlit render LaTeX between them.
        self.setup_green(monkeypatch, rss_xml(
            rss_item("Fermus Wanted $30 Billion on Just $51 Million [sic]")
        ))
        news_md = text_of(run_check().markdown)
        assert r"\$30 Billion on Just \$51 Million \[sic\]" in news_md

    def test_no_news_shows_yellow_manual_check(self, monkeypatch):
        self.setup_green(monkeypatch)  # autouse fixture: RSS fails
        warnings = text_of(run_check().warning)
        assert "לא נמצאו כותרות" in warnings
        assert "https://finance.yahoo.com/quote/NVDA/news" in warnings

    def test_veto_buttons_shown_and_risk_hidden_until_decision(self, monkeypatch):
        self.setup_green(monkeypatch)
        at = run_check()
        assert at.button(key="approve").label == "אשר והמשך לניהול סיכונים"
        assert at.button(key="veto").label == "פסול מניה"
        assert RISK_TITLE not in page_text(at)

    def test_approve_shows_capital_shield(self, monkeypatch):
        self.setup_green(monkeypatch)
        plan = tg.compute_risk_plan(tg.latest_snapshot(tg.compute_indicators(self.df)))
        at = run_check()
        at.button(key="approve").click()
        at.run()
        assert not at.exception
        text = page_text(at)
        assert RISK_TITLE in text
        assert "מתאימה להמשך בדיקה" in text  # report persisted across rerun
        assert "תנודתיות יומית ממוצעת (ATR)" in text
        assert f"{plan.atr:,.2f}" in text
        assert f"{plan.stop_loss:,.2f}" in text
        assert f"{plan.take_profit:,.2f}" in text
        assert f"{plan.shares:,}" in text
        # Expected profit is the last row of the details table.
        table = next(m.value for m in at.markdown if "| נתון | ערך |" in m.value)
        last_row = table.strip().splitlines()[-1]
        assert last_row == f"| רווח מתוכנן אם היעד יושג | \\${plan.expected_profit:,.2f} |"
        assert "פער מחירים (Gap)" in text_of(at.warning)
        assert {"approve", "veto"}.isdisjoint(button_keys(at))

    def approve_with(self, monkeypatch, capital, risk_pct):
        self.setup_green(monkeypatch)
        at = new_app()
        at.number_input(key="capital").set_value(capital)
        at.number_input(key="risk_pct").set_value(risk_pct)
        at.text_input(key="ticker").input("NVDA")
        at.button(key="check").click()
        at.run()
        at.button(key="approve").click()
        at.run()
        assert not at.exception
        return at

    def test_sidebar_values_drive_the_capital_shield(self, monkeypatch):
        at = self.approve_with(monkeypatch, capital=50_000, risk_pct=2.0)
        snap = tg.latest_snapshot(tg.compute_indicators(self.df))
        plan = tg.compute_risk_plan(snap, capital=50_000, risk_pct=0.02)
        text = page_text(at)
        assert r"\$50,000.00" in text   # capital (escaped '$' in markdown)
        assert r"\$1,000.00" in text    # 2% risk budget
        assert f"{plan.shares:,}" in [str(m.value) for m in at.metric]
        assert "(2%)" in text

    def test_fractional_risk_percent_is_displayed_exactly(self, monkeypatch):
        at = self.approve_with(monkeypatch, capital=10_000, risk_pct=1.5)
        assert "(1.5%)" in page_text(at)
        assert "גדול מ-1.5%" in text_of(at.warning)

    def test_changing_settings_after_approval_recalculates(self, monkeypatch):
        at = self.approve_with(monkeypatch, capital=10_000, risk_pct=1.0)
        snap = tg.latest_snapshot(tg.compute_indicators(self.df))
        at.number_input(key="capital").set_value(100_000)
        at.run()
        expected = tg.compute_risk_plan(snap, capital=100_000, risk_pct=0.01)
        assert RISK_TITLE in page_text(at)
        assert f"{expected.shares:,}" in [str(m.value) for m in at.metric]

    def test_veto_disqualifies_without_capital_shield(self, monkeypatch):
        self.setup_green(monkeypatch)
        at = run_check()
        at.button(key="veto").click()
        at.run()
        assert not at.exception
        assert "נפסלה לפי החלטתך" in text_of(at.error)
        assert RISK_TITLE not in page_text(at)
        assert {"approve", "veto"}.isdisjoint(button_keys(at))

    def test_new_check_resets_previous_decision(self, monkeypatch):
        self.setup_green(monkeypatch)
        at = run_check()
        at.button(key="approve").click()
        at.run()
        at.button(key="check").click()
        at.run()
        assert RISK_TITLE not in page_text(at)
        assert {"approve", "veto"} <= button_keys(at)

    def test_yellow_stock_also_gets_veto_step(self, monkeypatch):
        patch_market(monkeypatch, history_result=make_ohlcv(UPTREND))
        at = run_check()
        assert {"approve", "veto"} <= button_keys(at)
