# Stock Predictor — UI/UX Improvement Plan

Status: Draft · Owner: clpatel · Date: 2026-06-13

This plan addresses the visual and structural issues in the prediction results
view. Items are ordered by dependency: the market-status foundation must land
first because every header and stats decision branches off it.

---

## 0. Foundation — Market-Status Detection (blocker for everything below)

### Problem
`market_status()` in [utils.py:22-31](utils.py#L22-L31) only inspects the
*time of day*. It does **not** check the day of week or market holidays. On a
Saturday at noon it returns `MARKET_OPEN`; on a holiday Monday before 9:30 it
returns `BEFORE_MARKET_OPEN` as if Tuesday's session were imminent. Since the
competing-header fix and the stats-card label both branch on this function, it
has to be correct before any UI work.

### Target state — what the user sees: OPEN or CLOSED
The user only ever distinguishes **Open** vs **Closed**. Internally there are
three statuses, but the before/after split only matters to code, never to the
display. The single display rule: **when Closed, show the last completed
session's date (from the data); when Open, show live "Last Traded" with no
date.**

| Internal status | Condition | Market | Display |
|---|---|---|---|
| `BEFORE_MARKET_OPEN` | Trading day, now < 09:30 ET | Closed | Last session date + that day's final metrics |
| `MARKET_OPEN` | Trading day, 09:30 ≤ now ≤ 16:00 ET | Open | "Last Traded", live OHLC, no date |
| `AFTER_MARKET_CLOSE` | Trading day, now > 16:00 ET **or** any non-trading day (weekend/holiday) | Closed | Last session date + final metrics |

**Corrected logic (per review):** a non-trading day is `AFTER_MARKET_CLOSE`, not
`BEFORE_MARKET_OPEN` — on a weekend/holiday the last market event was a *close*.
This doesn't change what the user sees (still just "Closed", still the last
session date), but it keeps the status name honest. Friday-after-close, the
weekend, and a holiday Monday all read as Closed showing Friday's session.

### Approach
1. **Previous trading day comes from the data, not a calculation.** The last row
   of the downloaded frame (`data.index[-1].date()`) *is* the last completed
   session. Keep using that — already done in [render_ui.py:56](render_ui.py#L56).
2. **Holiday-aware trading-day check — `pandas_market_calendars` (chosen).** A
   pure clock cannot know Monday is Memorial Day. Use the NYSE calendar (`XNYS`)
   to ask whether *today* is a trading day. Offline, deterministic, no live
   network dependency:
   ```python
   import pandas_market_calendars as mcal
   nyse = mcal.get_calendar("XNYS")
   sched = nyse.schedule(start_date=today, end_date=today)
   is_trading_day = not sched.empty
   ```
   (yfinance 1.4.1 also exposes a live `Market("US").status`, but it's a network
   call and US-only — rejected in favor of the offline calendar.)
3. **Rewrite `market_status()`** to return one of `BEFORE_MARKET_OPEN`,
   `MARKET_OPEN`, `AFTER_MARKET_CLOSE` using *both* the calendar and the clock:
   - Not a trading day (weekend/holiday) → `AFTER_MARKET_CLOSE`.
   - Trading day, now < open → `BEFORE_MARKET_OPEN`.
   - Trading day, open ≤ now ≤ close → `MARKET_OPEN`.
   - Trading day, now > close → `AFTER_MARKET_CLOSE`.
4. Add `pandas_market_calendars` to [requirements.txt](requirements.txt).
5. Cache the calendar/schedule lookup per run so we don't rebuild it once per
   ticker.

**Status: DONE** ([utils.py](utils.py)) — `is_trading_day()` cached per date;
`market_status()` rewritten; covered by [tests/test_market_status.py](tests/test_market_status.py)
(18 cases, time injected via monkeypatch).

### Acceptance (locked by tests)
- Saturday/Sunday any time → Closed (`AFTER_MARKET_CLOSE`), showing Friday's data.
- Memorial Day (Mon) → Closed, showing Friday's data.
- Normal weekday 10:00 → Open (`MARKET_OPEN`), "Last Traded".
- Normal weekday 08:00 → Closed (`BEFORE_MARKET_OPEN`), showing prior session date.
- Normal weekday 17:00 → Closed (`AFTER_MARKET_CLOSE`), showing today's date.

---

## 1. Two Competing Headers → One Status-Driven Header

### Problem
Two stacked headers fight for the anchor:
[display_market_status.py:6](display_market_status.py#L6) renders a centered
date `<h2>`, then [render_ui.py:58](render_ui.py#L58) renders a left-aligned
`st.subheader("Today's Close Predictions")`. The status block and the section
title are redundant and visually disagree (centered vs left).

### Fix
- Collapse to a **single header** whose title is driven by `market_status()`:
  - Before Open → "Predictions for `<prev trading day>`"
  - After Close → "Today's Close Predictions"
  - Market Open → "Live — Last Traded"
- Keep the status icon (⏳ / 🔔 / 🔴 — already defined in
  [display_market_status.py:10-26](display_market_status.py#L10-L26)) inline with
  the title, not as a separate centered block.
- Remove the standalone `st.subheader("Today's Close Predictions")` at
  [render_ui.py:58](render_ui.py#L58); the header now carries that text.
- Pick one alignment (left) for the whole block so it stops fighting itself.

### Status note (subtitle) — one rule: Closed shows the session date
Collapse to a single binary on `market_status()`:
| Market | Subtitle |
|---|---|
| **Open** (`MARKET_OPEN`) | **None** — header says "Live — Last Traded"; the data is today's, self-explanatory. |
| **Closed** (`BEFORE_MARKET_OPEN` or `AFTER_MARKET_CLOSE`) | Show the **displayed session's date** from `data.index[-1].date()` (e.g. "Predictions for the last traded session — Friday, June 12"). Before vs after open does not change this. |

So the subtitle ([display_market_status.py:8-37](display_market_status.py#L8-L37))
renders **only when `market_status() != MARKET_OPEN`**, and its text is the
session date pulled from the data — never a hardcoded string.
- **Remove** the hardcoded "Today's closing prices are final." and the
  Market-Open subtitle strings at
  [display_market_status.py:14-24](display_market_status.py#L14-L24).

---

## 2. Prediction Strip Cramped → Two Model Cards

### Problem
`preds_sameline()` ([render_helpers.py:61-77](render_helpers.py#L61-L77)) jams
ticker + Linear Regression + XGBoost + both deltas onto one tight line inside a
single pill ([render_helpers.py:80-90](render_helpers.py#L80-L90)).

### Fix
- Split into **two side-by-side model cards** (Linear Regression | XGBoost).
- Each card: model name, predicted price (large), delta vs reference price.
- Reference price = Close (after close), Last Traded (open), or prev close
  (before open) — same source the strip uses now.
- Stack the two cards on narrow viewports (CSS grid `auto-fit`, same pattern as
  the stats grid at [render_helpers.py:231-232](render_helpers.py#L231-L232)).

---

## 3. Stats Cards → One Grouped Panel with Status-Aware Header

### Problem
The OHLC/Volume cards float disconnected below the chart with no header
([render_helpers.py:190-236](render_helpers.py#L190-L236)). They won't always
hold *today's* data (before open they hold the previous session).

### Fix
- Wrap the grid in **one bordered panel** with a header.
- Header text follows the **same binary as §1** — `market_status() == MARKET_OPEN`:
  - **Open** → "Today's Data" (live session).
  - **Closed** → the displayed session's date from `data.index[-1].date()`
    (e.g. "Friday, June 12"). Same date string the §1 subtitle uses.
- The session date is derived once from the data and reused by the §1 subtitle
  and this panel header — single source, no separate predicate.
- This removes the ambiguity about whose data the cards show.

---

## 4. Close / Last-Traded Card — Consistent Background, Colored Number

### Decision (per your call)
- Keep the card background **neutral/consistent** — do not tint the whole card
  red or green.
- Color **only the number**: green if up vs prev close, red if down, neutral if
  equal.
- Currently [render_helpers.py:194-221](render_helpers.py#L194-L221) tints the
  whole card background (`close_bg`) and border. Change so background/border are
  the same neutral as other cards; apply up/down/equal color to the value text
  only.
- Equal case: add the neutral branch (current code only has up/down).

---

## 5. Deltas — Bold

### Fix
- Make the `(+$3.29)` / `(-$0.68)` deltas **bold** and slightly larger.
- In `preds_sameline()` ([render_helpers.py:75](render_helpers.py#L75)) the delta
  span has color but no weight — add `font-weight: 700`. Carries over to the new
  model cards in §2.
- Up = green, down = red, equal = neutral (keep existing color map at
  [render_helpers.py:66](render_helpers.py#L66)).

---

## 6. Chart Chrome — Remove Noise

### Fix
- Remove the "TRADINGVIEW STYLE CHART" eyebrow label
  ([render_helpers.py:121](render_helpers.py#L121)).
- Remove the "OHLC candles from local market data" caption
  ([render_helpers.py:124](render_helpers.py#L124)) — everyone knows candles.
- Keep "`<TICKER>` · Last 10 Trading Days"
  ([render_helpers.py:122](render_helpers.py#L122)) as the only chart title.
- Keep the right-side Y-axis whitespace and the red dashed price line —
  TradingView defaults, intentionally kept for breathing room.

---

## 7. Visual Cohesion — Dark Mode for Cards + Edge Alignment

### Problem
The chart is dark ([render_helpers.py:118-140](render_helpers.py#L118-L140)) but
the prediction pill and stats cards are light
([render_helpers.py:83](render_helpers.py#L83),
[render_helpers.py:219](render_helpers.py#L219)). The light/dark mix is the core
of the "visually disturbing" feeling. Cards also don't align to the chart's
left/right edges.

### Decision — default dark, with a light-mode toggle
- **Default to full dark mode** for the prediction cards and stats panel to match
  the chart. One palette top to bottom reads as a single designed surface instead
  of three pasted widgets. Reuse the chart's tokens: bg `#0f172a`/`#111827`,
  borders `rgba(148,163,184,0.18)`, text `#f8fafc`/`#cbd5e1`.
- **Add a theme-toggle icon** (🌙 / ☀️) so the user can switch to light mode.
  - Store choice in `st.session_state` (e.g. `st.session_state.theme`,
    default `"dark"`); flip on click.
  - Drive card/panel colors from a small **token dict** keyed by theme rather
    than hardcoding hex inline — one `THEME["dark"]` / `THEME["light"]` map
    consumed by `display_predictions`, `create_grid_display`, and the chart
    builder. This is the prerequisite refactor; the toggle just selects the key.
  - The TradingView chart `layout.background`/`textColor`
    ([render_helpers.py:138-149](render_helpers.py#L138-L149)) reads from the
    same tokens so the chart flips with the rest.
- **Align all cards to the chart's edges** — same max width / horizontal padding
  as the chart container; no overflow except intentional responsive wrapping.
- The grouped stats panel (§3) inherits whichever theme is active.

---

## Suggested Implementation Order
1. **§0** Market-status calendar fix (unblocks §1, §3 labels). Add dependency.
2. **§1** Single status-driven header.
3. **§3** Grouped, status-aware stats panel.
4. **§2** Two prediction model cards.
5. **§4 + §5** Close-card number coloring + bold deltas.
6. **§6** Strip chart chrome.
7. **§7** Dark-mode palette + edge alignment (final visual pass over §1–§6).

## Resolved Decisions
- **Calendar lib:** `pandas_market_calendars` (`XNYS`). Offline, holiday-aware.
- **Status:** non-trading day → `AFTER_MARKET_CLOSE` (not Before Open). User sees
  only Open vs Closed.
- **Display binary:** `market_status() == MARKET_OPEN`. Open → live "Last Traded",
  no date. Closed (before/after/weekend/holiday) → show last session's date from
  `data.index[-1].date()`. Drives both the §1 subtitle and the §3 panel header.
- **Theme:** default full dark + a light-mode toggle icon, driven by a theme
  token dict in session state.
- **Half-days:** out of scope — keep hardcoded 09:30/16:00.

## Testing Strategy — every change tested in isolation
The cross-branch risk ("if one change affects another, how would you know?") is
answered by a **committed `pytest` suite**, not ad-hoc checks:
- **One test file per concern**, named after the section it locks
  (`tests/test_market_status.py` for §0; later `test_header.py`,
  `test_stats_panel.py`, ...). Each file asserts only its own section's
  behavior — that is the "in isolation" part.
- **Determinism:** anything time- or environment-dependent is injected, not
  read live. §0 monkeypatches `utils.get_nyse_datetime` so the same 18 cases
  pass regardless of when the suite runs.
- **Regression net across branches:** every feature branch runs the **full**
  suite before merge. If a styling change in §2 silently alters status output,
  §0's tests fail on that branch — that's how we know. Pure-CSS sections that
  emit HTML get a smoke test asserting the key tokens/strings are present.
- **Gate:** `pytest -q` green is a merge precondition for each branch.
- Add `pytest` to [requirements.txt](requirements.txt) (done).
