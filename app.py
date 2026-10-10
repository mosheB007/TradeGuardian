"""TradeGuardian - V0 web GUI (Streamlit).

A browser front-end for the same educational flow as the CLI
(tg_mentor_mvp.py). It renders Hebrew right-to-left correctly and works on
mobile screens.

All business logic - data fetching, indicators, the 3-level filter, the
earnings blackout, RSS news and the capital shield - is imported unchanged
from tg_mentor_mvp.py, so the CLI, the GUI and the test-suite share one
implementation. This file only contains presentation and UI state.

Flow (same as the CLI):
    1. Fetch prices. Earnings due today..+14 days -> red block, stop.
       (Earnings date unknown -> yellow warning, continue.)
    2. Technical report + level box (success / warning / error). RED -> stop.
    3. GREEN/YELLOW: headlines + "approve" / "disqualify" buttons.
       Approve -> capital shield.  Disqualify -> red veto message.

Run:
    streamlit run app.py
    # phone on the same Wi-Fi:  streamlit run app.py --server.address 0.0.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from urllib.parse import quote

import streamlit as st

from tg_mentor_mvp import (
    EARNINGS_BLACKOUT_DAYS,
    EARNINGS_UNKNOWN_MESSAGE,
    REWARD_RISK_RATIO,
    RISK_PER_TRADE,
    RSI_THRESHOLD,
    STOP_ATR_MULTIPLIER,
    YAHOO_NEWS_URL,
    Level,
    NewsItem,
    RiskPlan,
    Snapshot,
    VIRTUAL_CAPITAL,
    TradeGuardianError,
    classify,
    compute_indicators,
    compute_risk_plan,
    drop_incomplete_session,
    fetch_news,
    fetch_next_earnings_date,
    fetch_price_history,
    is_earnings_too_close,
    latest_snapshot,
    normalize_ticker,
    tradingview_url,
)

# ---------------------------------------------------------------------------
# Page setup & RTL
# ---------------------------------------------------------------------------

RTL_CSS = """
<style>
/* RTL only for content. The page frame stays LTR: on mobile Streamlit hides
   the collapsed sidebar with translateX(-width), which assumes the sidebar
   sits on the left; an RTL frame would drag it over the main content. */
[data-testid="stMain"], [data-testid="stSidebarUserContent"] {
    direction: rtl;
    text-align: right;
}
/* Streamlit pins text-align:left on headings and markdown/element
   containers, which would otherwise push Hebrew headings to the left. */
[data-testid="stHeading"], [data-testid="stHeading"] *,
[data-testid="stMarkdown"], [data-testid="stMarkdownContainer"],
[data-testid="stElementContainer"], [data-testid="stCaptionContainer"] {
    text-align: right !important;
}
/* Lists: put the bullet/number gutter on the right-hand side. */
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol {
    padding-right: 1.5em;
    padding-left: 0;
}
/* Tickers, numbers and code read left-to-right. */
input, code, pre {
    direction: ltr;
    text-align: left;
}
[data-testid="stMetricValue"] {
    direction: ltr;
}
/* Signed price deltas: in RTL '-$0.43 (-0.28%)' would read '(0.28%-) $0.43-'. */
[data-testid="stMetricDelta"] {
    direction: ltr;
}
</style>
"""

DISCLAIMER = (
    "הבהרה: המערכת היא כלי לימודי בלבד ואינה מספקת איתותי מסחר או ייעוץ השקעות. "
    "ההחלטה והאחריות הן שלך בלבד."
)

APPROVE, VETO = "approve", "veto"


# ---------------------------------------------------------------------------
# Analysis (mirrors tg_mentor_mvp.main(), without printing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    """Everything the page needs to render one check; kept in session_state."""

    ticker: str
    today: date
    session_open: bool
    earnings: date | None
    snapshot: Snapshot | None = None  # None when stopped by the earnings blackout
    level: Level | None = None
    news: tuple[NewsItem, ...] = ()

    @property
    def earnings_blocked(self) -> bool:
        return self.earnings is not None and is_earnings_too_close(
            self.earnings, self.today
        )


def run_analysis(raw_ticker: str) -> Analysis:
    """Run the CLI pipeline for one ticker. Raises TradeGuardianError."""
    ticker = normalize_ticker(raw_ticker)
    history = fetch_price_history(ticker)
    history, session_open = drop_incomplete_session(history)

    today = date.today()
    earnings = fetch_next_earnings_date(ticker, today)
    result = Analysis(ticker, today, session_open, earnings)
    if result.earnings_blocked:
        return result  # PRD 3.D: stop before any technical analysis

    snapshot = latest_snapshot(compute_indicators(history))
    level = classify(snapshot)
    news = tuple(fetch_news(ticker)) if level is not Level.RED else ()
    return Analysis(ticker, today, session_open, earnings, snapshot, level, news)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _md(text: str) -> str:
    """Escape markdown/LaTeX control characters in external text (headlines).

    '$' matters most: Streamlit renders '$...$' as math, which would mangle
    headlines such as '$30 Billion Valuation on Just $51 Million'.
    """
    for ch in "\\`*_[]<>$#|":
        text = text.replace(ch, "\\" + ch)
    return text


def _url(url: str) -> str:
    """Make a URL safe inside markdown link parentheses."""
    return quote(url, safe=":/?&=%#.,~+-_@!;'")


def _money(value: float) -> str:
    return f"\\${value:,.2f}"  # escaped: a bare '$' pair would render as LaTeX


def _pct(fraction: float) -> str:
    """0.015 -> '1.5%', 0.01 -> '1%' (the sidebar allows 0.1% steps)."""
    return f"{round(fraction * 100, 2):g}%"


def _days_until_text(days: int) -> str:
    if days == 0:
        return "היום"
    if days == 1:
        return "מחר"
    return f"בעוד {days} ימים"


def render_earnings(a: Analysis) -> None:
    if a.earnings is None:
        st.warning(EARNINGS_UNKNOWN_MESSAGE, icon="⚠️")
        return
    when = _days_until_text((a.earnings - a.today).days)
    if a.earnings_blocked:
        st.error(
            f"**נפסלת - דוח כספי קרוב: {a.ticker}**\n\n"
            f"החברה צפויה לפרסם דוח כספי בתאריך {a.earnings.isoformat()} ({when}).\n"
            f"זה בתוך חלון של {EARNINGS_BLACKOUT_DAYS} הימים הקרובים - "
            "קרוב מדי ומסוכן לסחור.\n\n"
            "סביב פרסום דוח כספי המחיר עלול לזנק או לצנוח בפער (Gap) חד בבת אחת, "
            "וקו ההגנה (Stop-Loss) לא יוכל להגן עליך.\n\n"
            "הניתוח נעצר כאן. בדוק את המניה שוב לאחר פרסום הדוח.",
            icon="🛑",
        )
        return
    st.info(
        f"תאריך הדוח הכספי הבא: {a.earnings.isoformat()} ({when}) - "
        f"מחוץ לחלון הסיכון של {EARNINGS_BLACKOUT_DAYS} ימים.",
        icon="📅",
    )


def _daily_change_delta(s: Snapshot) -> tuple[str | None, bool]:
    """Daily-change delta text for st.metric, e.g. '+$1.23 (+0.55%)'.

    Returns (delta, is_flat). Streamlit colors a string delta red when it
    starts with '-', else green; a flat day should be shown gray, arrowless.
    """
    change, pct = s.daily_change, s.daily_change_pct
    if change is None or pct is None:
        return None, False
    flat = round(change, 2) == 0
    sign = "-" if change < 0 and not flat else "+"
    return f"{sign}${abs(change):,.2f} ({0.0 if flat else pct:+.2f}%)", flat


def render_indicators(s: Snapshot) -> None:
    volume_pct = (s.volume / s.volume_sma_20 * 100) if s.volume_sma_20 else 0.0
    c1, c2, c3 = st.columns(3)
    daily_delta, flat_day = _daily_change_delta(s)
    c1.metric(
        "מחיר סגירה אחרון",
        f"{s.close:,.2f}",
        daily_delta,
        delta_color="off" if flat_day else "normal",
        delta_arrow="off" if flat_day else "auto",
    )
    c2.metric(
        "ממוצע נע 50 יום (SMA 50)",
        f"{s.sma_50:,.2f}",
        "המחיר מעל הממוצע" if s.above_sma else "המחיר מתחת/על הממוצע",
        delta_color="normal" if s.above_sma else "inverse",
        delta_arrow="up" if s.above_sma else "down",
    )
    c3.metric("מדד RSI (14 ימים)", f"{s.rsi_14:.1f}")
    c4, c5 = st.columns(2)
    c4.metric("מחזור מסחר אחרון", f"{s.volume:,.0f}")
    c5.metric(
        "ממוצע מחזור 20 יום",
        f"{s.volume_sma_20:,.0f}",
        f"{volume_pct:.0f}% מהממוצע",
        delta_color="off",
        delta_arrow="off",
    )


def render_level(s: Snapshot, level: Level) -> None:
    if level is Level.GREEN:
        st.success(
            "**סיווג: מתאימה להמשך בדיקה (ירוק)**\n\n"
            "המגמה חיובית - המחיר נמצא מעל ממוצע 50. "
            f"יש ירידה במחיר (RSI = {s.rsi_14:.1f}, מתחת ל-{RSI_THRESHOLD:.0f}), "
            "ללא פאניקת מוכרים (מחזור המסחר האחרון נמוך מהממוצע).\n\n"
            "**המשימה שלך: פתח את הגרף וחפש עצירת מחיר או נר היפוך.**\n"
            "1. **עצירת מחיר** - האם הירידה נעצרת? חפש כמה נרות קטנים ברצף, או נגיעה "
            "בקו ממוצע 50 / באזור תמיכה (שפל קודם) שממנו המחיר לא ממשיך לרדת.\n"
            "2. **נר היפוך** - נר עם 'זנב' תחתון ארוך וגוף קטן (פטיש / Hammer), "
            "או נר ירוק גדול ש'בולע' את הנר האדום שלפניו (Bullish Engulfing).\n"
            "3. **מחזור מסחר** - ודא שבנרות הירידה המחזור נמוך (אין מכירה המונית), "
            "ובנר ההיפוך המחזור עולה (קונים חוזרים לשוק).\n\n"
            "כל עוד לא ראית עצירה או נר היפוך ברור - אין מה לעשות. סבלנות.",
            icon="🟢",
        )
    elif level is Level.YELLOW:
        reasons = []
        if not s.rsi_is_low:
            reasons.append(
                f"- RSI = {s.rsi_14:.1f} ({RSI_THRESHOLD:.0f} ומעלה): המניה 'יקרה' יחסית - "
                "היא עדיין לא ירדה מספיק לנקודת כניסה נוחה."
            )
        if not s.volume_is_calm:
            reasons.append(
                "- מחזור המסחר האחרון גבוה מהממוצע של 20 יום: "
                "יש תנודתיות / פעילות חריגה."
            )
        st.warning(
            "**סיווג: דורשת זהירות (צהוב)**\n\n"
            "המגמה חיובית - המחיר נמצא מעל ממוצע 50, "
            "אך התזמון אינו אידיאלי כרגע. הסיבה:\n"
            + "\n".join(reasons)
            + "\n\n**המשימה שלך: להמתין להזדמנות טובה יותר.**\n"
            "1. האם המחיר 'מתוח' הרחק מעל קו ממוצע 50? אם כן - המתן לתיקון (ירידה) "
            "לכיוון הממוצע לפני שבכלל שוקלים כניסה.\n"
            "2. הסתכל על עמודות הווליום מתחת לגרף: האם יש עמודה חריגה בגובהה? "
            "האם היא בנר אדום (לחץ מכירות) או ירוק (קנייה נלהבת)? בשני המקרים - "
            "לא נכנסים ברגע של תנודתיות חריגה.\n"
            "3. הוסף את המניה לרשימת מעקב ובדוק שוב בעוד כמה ימים - כאשר ה-RSI ירד "
            f"מתחת ל-{RSI_THRESHOLD:.0f} והמחזור יירגע אל מתחת לממוצע.",
            icon="🟡",
        )
    else:
        st.error(
            "**סיווג: נפסלת (אדום)**\n\n"
            "המגמה שלילית - המחיר נמצא מתחת לממוצע הנע 50 (או בדיוק עליו).\n\n"
            "**המשימה שלך: אין לחפש עסקאות קנייה במניה זו כרגע. "
            "עדיף לשבת על הגדר.**\n"
            "1. ראה כיצד המחיר נמצא מתחת לקו ממוצע 50 - הקו משמש כעת 'תקרה' (התנגדות).\n"
            "2. שים לב לרצף של שיאים נמוכים יותר ושפלים נמוכים יותר - סימן למגמת ירידה.\n"
            "3. אל תנסה 'לתפוס סכין נופלת'. המניה תחזור לשיקול רק כשהמחיר יחזור "
            "ויתבסס מעל ממוצע 50.",
            icon="🔴",
        )


def render_chart_link(ticker: str) -> None:
    # English kept at the end of the sentence so the RTL line reads naturally.
    st.markdown(f"[לחץ כאן לפתיחת הגרף המלא ב-TradingView]({tradingview_url(ticker)})")
    with st.expander("איך להגדיר את הגרף?"):
        st.markdown(
            "**הוראות הגדרה ב-TradingView:**\n"
            "1. לחץ למעלה על כפתור 'Indicators' (סמל של fx).\n"
            "2. חפש 'SMA' (Moving Average Simple), לחץ עליו, כנס להגדרות שלו "
            "(גלגל שיניים) ושנה את ה-Length ל-50.\n"
            "3. חפש 'RSI' (Relative Strength Index) ולחץ עליו.\n\n"
            "כעת אתה אמור לראות את קו הממוצע על הגרף, "
            "ואת ה-RSI והמחזורים (Volume) בתחתית המסך."
        )


def render_news(ticker: str, items: tuple[NewsItem, ...]) -> None:
    # The English term sits on its own line: mixed inside a Hebrew heading
    # its parentheses get reordered by the bidi algorithm.
    st.subheader("חדשות ואזהרות")
    st.caption("Human-in-the-Loop")
    if not items:
        st.warning(
            f"לא נמצאו כותרות חדשות עבור {ticker} (ייתכן שהשירות אינו זמין כרגע).\n\n"
            f"בדוק ידנית: {YAHOO_NEWS_URL.format(symbol=ticker)}",
            icon="⚠️",
        )
    for i, item in enumerate(items, start=1):
        title = _md(item.title)
        headline = f"[{title}]({_url(item.url)})" if item.url else title
        meta = f"  \n{item.published}" if item.published else ""
        st.markdown(f"{i}. **{headline}**{meta}")
    st.caption(
        "עבור על הכותרות (ופתח קישורים לפי הצורך) וחפש אירוע חריג, למשל: "
        "פשיטת רגל, חקירה רשמית, הנפקה מדללת, עזיבה פתאומית של מנכ\"ל, תביעה מהותית. "
        "חדשות רגילות (דירוג אנליסטים, השקת מוצר) אינן סיבה לפסילה."
    )


def _set_decision(decision: str) -> None:
    st.session_state.decision = decision


def render_veto_buttons() -> None:
    st.markdown("**האם זיהית בחדשות אירוע חריג שדורש פסילה?**")
    c1, c2 = st.columns(2)
    c1.button(
        "אשר והמשך לניהול סיכונים", key=APPROVE, type="primary", width="stretch",
        on_click=_set_decision, args=(APPROVE,),
    )
    c2.button(
        "פסול מניה", key=VETO, width="stretch",
        on_click=_set_decision, args=(VETO,),
    )


def render_veto_result(ticker: str) -> None:
    st.error(
        f"**נפסלת - {ticker} נפסלה לפי החלטתך על בסיס החדשות.**\n\n"
        "מגן ההון לא יחושב. זו בדיוק המטרה של זכות הווטו: "
        "עדיף לפספס עסקה מאשר להיכנס לאירוע לא צפוי.",
        icon="🛑",
    )


def render_risk(plan: RiskPlan) -> None:
    pct = _pct(plan.risk_pct)
    st.subheader("מגן הון (ניהול סיכונים לעסקה מדומה)")
    if not plan.is_valid:
        st.markdown(
            f"- **הון וירטואלי:** {_money(plan.capital)}\n"
            f"- **תנודתיות יומית ממוצעת (ATR):** {plan.atr:,.2f}"
        )
        st.error("לא ניתן לחשב תוכנית סיכון: ה-ATR הוא אפס או שמחיר הסטופ-לוס יוצא שלילי.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("קו הגנה (Stop-Loss)", f"{plan.stop_loss:,.2f}")
        c2.metric(f"יעד רווח (1:{REWARD_RISK_RATIO:g})", f"{plan.take_profit:,.2f}")
        c3.metric("כמות מניות מקסימלית", f"{plan.shares:,}")
        st.markdown(
            "| נתון | ערך |\n|---|---|\n"
            f"| הון וירטואלי | {_money(plan.capital)} |\n"
            f"| סיכון מקסימלי לעסקה ({pct}) | {_money(plan.risk_budget)} |\n"
            f"| תנודתיות יומית ממוצעת (ATR) | {plan.atr:,.2f} |\n"
            f"| מחיר כניסה (סגירה אחרונה) | {plan.entry:,.2f} |\n"
            f"| קו הגנה (Stop-Loss) - מרווח של פי {STOP_ATR_MULTIPLIER:g} מהתנודתיות "
            f"| {plan.stop_loss:,.2f} |\n"
            f"| הפסד אפשרי על כל מניה בודדת | {plan.risk_per_share:,.2f} |\n"
            f"| יעד רווח (Take-Profit), יחס 1:{REWARD_RISK_RATIO:g} "
            f"| {plan.take_profit:,.2f} |\n"
            f"| כמות מניות מקסימלית לקנייה | {plan.shares:,} |\n"
            f"| עלות הפוזיציה | {_money(plan.position_value)} |\n"
            f"| הפסד מתוכנן אם הסטופ יופעל | {_money(plan.max_loss)} |\n"
            f"| רווח מתוכנן אם היעד יושג | {_money(plan.expected_profit)} |\n"
        )
        if plan.capped_by_capital:
            st.info(
                "הכמות הוגבלה לפי ההון הווירטואלי (ללא מינוף), "
                f"ולכן הסיכון בפועל נמוך מ-{pct}."
            )
        elif plan.shares == 0:
            st.warning(
                "ההפסד האפשרי על כל מניה בודדת גבוה מהתקציב בסיכון - "
                f"לא ניתן לקנות אפילו מניה אחת בלי לחרוג מ-{pct}."
            )
    st.warning(
        "שים לב: זהו תכנון מתמטי לימודי. במקרה של פער מחירים (Gap) בפתיחת המסחר, "
        f"ההפסד בפועל עלול להיות גדול מ-{pct}",
        icon="⚠️",
    )


def render_analysis(a: Analysis, capital: float, risk_pct: float) -> None:
    if a.session_open:
        st.info(
            "המסחר היום עדיין פתוח והנתונים של היום חלקיים, "
            "ולכן הניתוח מבוסס על יום המסחר המלא האחרון.",
            icon="⏳",
        )
    render_earnings(a)
    if a.snapshot is None or a.level is None:
        return  # stopped by the earnings blackout

    st.subheader(f"דוח לימודי טכני: {a.ticker}")
    st.caption(f"יום המסחר האחרון בנתונים: {a.snapshot.date.isoformat()}")
    render_indicators(a.snapshot)
    render_level(a.snapshot, a.level)
    render_chart_link(a.ticker)
    if a.level is Level.RED:
        return

    st.divider()
    render_news(a.ticker, a.news)
    decision = st.session_state.get("decision")
    if decision == APPROVE:
        st.divider()
        # Recomputed on every rerun, so sidebar changes apply immediately.
        render_risk(compute_risk_plan(a.snapshot, capital=capital, risk_pct=risk_pct))
    elif decision == VETO:
        render_veto_result(a.ticker)
    else:
        render_veto_buttons()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[float, float]:
    """Virtual-portfolio settings. Returns (capital in USD, risk as a fraction)."""
    with st.sidebar:
        st.header("הגדרות תיק וירטואלי")
        capital = st.number_input(
            "הון וירטואלי בדולרים",
            min_value=1000.0,
            value=float(VIRTUAL_CAPITAL),
            step=1000.0,
            format="%.0f",
            key="capital",
        )
        risk_percent = st.number_input(
            "סיכון מקסימלי לעסקה (%)",
            min_value=0.1,
            max_value=10.0,
            value=RISK_PER_TRADE * 100,
            step=0.1,
            format="%.1f",
            key="risk_pct",
        )
        st.caption("השינויים חלים מיד על חישוב מגן ההון.")
    return float(capital), float(risk_percent) / 100


def main() -> None:
    st.set_page_config(page_title="TradeGuardian", page_icon="🛡️", layout="centered")
    st.markdown(RTL_CSS, unsafe_allow_html=True)

    st.title("🛡️ TradeGuardian")
    st.caption("המורה הטכני למסחר סווינג (V0) - כלי לימודי למתחילים")
    capital, risk_pct = render_sidebar()

    st.text_input("סימול מניה", key="ticker", placeholder="למשל NVDA")
    if st.button("בדוק מניה", key="check", type="primary", width="stretch"):
        st.session_state.decision = None
        st.session_state.analysis = None
        st.session_state.error = None
        try:
            with st.spinner("שולף נתוני מחיר וחדשות..."):
                st.session_state.analysis = run_analysis(st.session_state.ticker)
        except TradeGuardianError as exc:
            st.session_state.error = str(exc)
        except Exception:  # noqa: BLE001 - never show a traceback to the user
            st.session_state.error = (
                "אירעה שגיאה לא צפויה בעיבוד הנתונים. נסה שוב או בדוק טיקר אחר."
            )

    error = st.session_state.get("error")
    analysis = st.session_state.get("analysis")
    if error:
        st.error(error, icon="❌")
    elif analysis is not None:
        # Not isinstance(): Streamlit re-executes this script on every
        # interaction, so `Analysis` is a new class object on each rerun.
        render_analysis(analysis, capital, risk_pct)

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":  # `streamlit run` executes the script as __main__
    main()
