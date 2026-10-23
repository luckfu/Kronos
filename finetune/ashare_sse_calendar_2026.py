"""Shanghai Stock Exchange sessions used to freeze the 2026 selection calendar.

The holiday list is the SSE notice 上证公告〔2025〕45号, published 2025-12-22:

https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml

Weekends are closed. That notice does not schedule a weekend trading session
inside 2026. The 5月9日 line only restates that Saturday as a weekend closure.
The same notice closes 2026-09-25..09-27 for Mid-Autumn, which matches the
sealed window's missing 09-25. Do not invent sessions from ``pandas`` business
days: those keep Qingming Monday, Labour Day, and Dragon Boat Friday.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta


# Weekday closures on top of Saturday/Sunday. Inclusive ranges from the notice,
# with the weekend days left out of this set because ``weekday() >= 5`` already
# drops them.
_EXTRA_CLOSED = frozenset({
    # 元旦: 1月1日–1月3日. 1月3日 is Saturday.
    date(2026, 1, 1),
    date(2026, 1, 2),
    # 春节: 2月15日–2月23日. 2月15日 is Sunday; 2月21日–22日 are the weekend.
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
    date(2026, 2, 23),
    # 清明: 4月4日–4月6日. 4月4日–5日 are the weekend.
    date(2026, 4, 6),
    # 劳动节: 5月1日–5月5日. 5月2日–3日 are the weekend.
    date(2026, 5, 1),
    date(2026, 5, 4),
    date(2026, 5, 5),
    # 端午: 6月19日–6月21日. 6月20日–21日 are the weekend.
    date(2026, 6, 19),
    # 中秋: 9月25日–9月27日. 9月26日–27日 are the weekend.
    date(2026, 9, 25),
    # 国庆: 10月1日–10月7日.
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
    date(2026, 10, 7),
})

SELECTION_START = "2026-04-01"
SELECTION_END = "2026-06-30"
# Weekdays inside the selection window that pandas bdate_range would keep and
# the SSE notice closes. Locked so a business-day generator cannot sneak in.
SELECTION_HOLIDAY_WEEKDAYS = (
    "2026-04-06",
    "2026-05-01",
    "2026-05-04",
    "2026-05-05",
    "2026-06-19",
)


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def is_sse_session(value) -> bool:
    """True when ``value`` is an SSE session under the 2026 holiday notice."""
    day = _as_date(value)
    if day.weekday() >= 5:
        return False
    return day not in _EXTRA_CLOSED


def sessions_between(start, end) -> list[str]:
    """Inclusive ascending ``YYYY-MM-DD`` SSE sessions. Empty if ``end < start``."""
    first = _as_date(start)
    last = _as_date(end)
    if last < first:
        return []
    days = []
    cursor = first
    while cursor <= last:
        if is_sse_session(cursor):
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def selection_window_sessions() -> list[str]:
    """Trading days inside 2026-04-01..2026-06-30. The frozen Rank IC calendar."""
    return sessions_between(SELECTION_START, SELECTION_END)
