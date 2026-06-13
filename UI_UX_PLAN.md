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

### Target state — the three scenarios
| Scenario | Condition | Header / data shown |
|---|---|---|
| **1.1 Before Open** | Today is a trading day, now < 09:30 ET, **or** today is a non-trading day (weekend/holiday) and the next session hasn't opened | Previous trading day's date + that day's final metrics. Predictions framed as "for `<prev trading day>`". |
| **1.2 After Close** | Today is a trading day, now > 16:00 ET | "Today's Close Predictions". Today's final OHLC. |
| **2. Market Open** | Today is a trading day, 09:30 ≤ now ≤ 16:00 ET | "Last Traded Price" (logic already present in [render_helpers.py:204](render_helpers.py#L204)). Live/last OHLC. |

Note the weekend/long-weekend collapse: Friday after close, all of Saturday,
Sunday, and Monday-before-bell are the **same** state — "Before Open, showing
last completed session." A holiday Monday extends that window through Tuesday's
bell automatically once the calendar is consulted.

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
   - Not a trading day (weekend/holiday) → `BEFORE_MARKET_OPEN` (showing last
     session).
   - Trading day, now < open → `BEFORE_MARKET_OPEN`.
   - Trading day, open ≤ now ≤ close → `MARKET_OPEN`.
   - Trading day, now > close → `AFTER_MARKET_CLOSE`.
4. Add `pandas_market_calendars` to [requirements.txt](requirements.txt).
5. Cache the calendar/schedule lookup per run so we don't rebuild it once per
   ticker.

### Acceptance
- Saturday/Sunday any time → Before Open, showing Friday's data.
- Memorial Day (Mon) → Before Open through Tuesday 09:30, showing Friday's data.
- Normal weekday 10:00 → Market Open, "Last Traded".
- Normal weekday 17:00 → After Close, "Today's Close Predictions".

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

### Status note (subtitle) — shown ONLY in Before Open
The day has **three statuses** across **two market on/off states**:
| Status | Market | Note |
|---|---|---|
| Before Open | closed | **Show note** — "Predictions are for the previously traded date — `<prev trading day>`." |
| Market Open | open | **No note** |
| After Close | closed | **No note** |

So the subtitle ([display_market_status.py:8-37](display_market_status.py#L8-L37))
renders the note **only when `market_status() == BEFORE_MARKET_OPEN`**. In both
Open and After Close the data shown is today's, so the header title alone
("Live — Last Traded" / "Today's Close Predictions") is self-explanatory — no
subtitle.
- **Remove** the hardcoded "Today's closing prices are final." and the
  Market-Open subtitle strings at
  [display_market_status.py:14-24](display_market_status.py#L14-L24); only the
  Before-Open branch keeps a subtitle.

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
- Header text keyed on the **`data_is_from_today` predicate** (shared with §1's
  subtitle), NOT on a three-way status split. The rule is simply *is the data on
  screen from today or not*:
  - **Data IS from today** (Market Open *or* After Close — both show today's
    session) → "Today's Data". Market Open mid-session is still today.
  - **Data NOT from today** (Before Open / weekend / long weekend) → show the
    **date** of the session displayed (e.g. "Friday, June 12").
- One predicate computed once per run; reused by header subtitle (§1), this
  panel header, and anywhere else the displayed-session date matters.
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
- **Theme:** default full dark + a light-mode toggle icon, driven by a theme
  token dict in session state.
- **Status note:** rendered **only in Before Open**; Open and After Close show
  no subtitle (data is today's, header title suffices).
- **Stats panel header:** "Today's Data" when Open or After Close; the displayed
  session's date when Before Open. Same `data_is_from_today` predicate.
- **Half-days:** out of scope — keep hardcoded 09:30/16:00.
