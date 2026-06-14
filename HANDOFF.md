# HANDOFF — session handoff

Updated: 2026-06-13 · Branch: `main` @ `6b700ad` · Working tree: clean

## What this work is
Executing the UI/UX rework in [UI_UX_PLAN.md](UI_UX_PLAN.md) — fixing the
"visually disturbing" prediction results view (Streamlit app). Sections §0–§7,
done one per branch, dependency-ordered. Read UI_UX_PLAN.md first; its
**Progress** table is the source of truth for what's done.

## ⚠️ Workflow rule (NEW — honor it)
**Manually test each feature/fix in the running app BEFORE commit/PR/merge.**
After implementing, launch the app, give the user the localhost URL + the
expected render, and **wait for their "looks good"** before `git commit` / PR /
merge. Don't batch commit+PR+merge until they've eyeballed it. (Several §1 style
rounds — center vs left, subtitle size, gap — were driven by live render.)

## App shape (orientation)
- Streamlit app. Entry: [stock_predictors.py](stock_predictors.py)
  (`streamlit run stock_predictors.py`).
- Results UI: [render_ui.py](render_ui.py) `display_results()` →
  [render_helpers.py](render_helpers.py) (prediction pill, TradingView chart,
  stats grid) + [display_market_status.py](display_market_status.py) (header).
- Market status: [utils.py](utils.py) `market_status()` / `is_trading_day()` /
  `next_trading_day()`.
- venv: `./venv/Scripts/python.exe`, `./venv/Scripts/streamlit`.
- Tests: `./venv/Scripts/python.exe -m pytest tests/ -q` (currently 30 passing).

## Done (merged to main)
- **§0** holiday/weekend-aware `market_status()` (PR #8) + `MARKET_NOW` env
  override (PR #9) + gitignore `*_data.csv`. Tests
  [tests/test_market_status.py](tests/test_market_status.py) (18 cases).
- **§1** single status-driven header (PR #10) + style polish (PR #11, #12).
  Implemented in [display_market_status.py](display_market_status.py):
  - **Centered, two-line** header via `generate_market_status_header()`:
    title line (status phrase + icon), grey subtitle below (~65% size, snug,
    `margin-top: -10px`). States:
    - `BEFORE_MARKET_OPEN` → 🔴 "Market Closed" + "Displaying Predictions for
      `<last session>`".
    - `MARKET_OPEN` → 🔔 "Live — Last Traded" (no subtitle).
    - `AFTER_MARKET_CLOSE` trading day → 🔴 "Today's Close Predictions and
      Actuals" (no subtitle).
    - `AFTER_MARKET_CLOSE` weekend/holiday → 🔴 "Market Closed Weekend/Holiday"
      + "Displaying Predictions for `<last session>`".
    - "Displaying" = deliberate "not today, past-session accuracy recap" cue.
      After-close split via `is_trading_day(today)`.
  - **Dated next-day section** via `generate_next_day_header()` →
    "Predictions for `<next_session>`", `next_session =
    utils.next_trading_day(last session)` (XNYS; before bell = today). Replaced
    generic `st.subheader("Next Day's Close Predictions")` in render_ui.py.
  - Tests [tests/test_header.py](tests/test_header.py) (12 cases).
  - Verified manually with `MARKET_NOW` (before-open Mon + weekend Sat).

## Key domain fact (verified in pipeline this session)
Today-target models use **today's intraday OHLCV** as features
([feature_engineering.py:179](feature_engineering.py#L179)), fed from the last
fetched row ([pipeline.py:57-58,128](pipeline.py#L57-L58)). **Before the bell
there is no today row**, so the "today" prediction is really a re-prediction of
the **last** session. The real forward prediction (targets today) is the
**next-day** model (its features include `Close`,
[feature_engineering.py:181](feature_engineering.py#L181)). This is why before
open the main section is framed as a past-session accuracy recap.

## Next: §3 — grouped, status-aware stats panel
Branch `feat/stats-panel` off fresh `main`. Plan §3 in UI_UX_PLAN.md. Summary:
- Wrap the floating OHLC/Volume cards
  ([render_helpers.py:190-236](render_helpers.py#L190-L236)) in ONE bordered
  panel with a header.
- Panel header follows the same binary as §1: `market_status() == MARKET_OPEN`
  → "Today's Data"; Closed → the displayed session date from
  `data.index[-1].date()` (reuse the same date the §1 subtitle shows).
- Add `tests/test_stats_panel.py` (smoke-test rendered HTML strings per status).

## ⚠️ Open follow-up (own branch, before/after §3 — user's call)
**Before-open prediction-vs-actual data wiring.** §1 fixed header *labels* only.
Before open the main section's numbers are the today-model run on the last row
(a re-prediction of the last session). To make the "Displaying Predictions for
`<Friday>`" panel actually pair **Friday's prediction vs Friday's actual close**,
change [render_ui.py](render_ui.py) `display_results()` to feed the last-session
prediction + actual to the main section when `market_status() != MARKET_OPEN`.
See UI_UX_PLAN.md §1 "Follow-up".

## Decisions locked (see UI_UX_PLAN.md "Resolved Decisions")
- Calendar lib: `pandas_market_calendars` (XNYS), offline.
- User sees only Open/Closed; Closed shows last session date.
- "Displaying" subtitle cue only on closed-with-past-session states.
- Theme (§7): default full dark + light-toggle icon, via a theme token dict.
- Half-days: out of scope (hardcoded 09:30/16:00).

## Workflow / conventions
- One branch per section: branch off main → implement → **manual test (see rule
  above)** → tests → `pytest -q` green → push → `gh pr create` → `gh pr merge
  --merge --delete-branch`.
- Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Each section gets its own isolated test file.
- Docs/PROGRESS updates go direct to main.

## Gotchas
- Status only renders after clicking **Predict** (runs `yf.download` — live
  network, can lag/flake).
- Manual UI states via `MARKET_NOW`, e.g.
  `$env:MARKET_NOW="2026-06-13T12:00"; ./venv/Scripts/streamlit run stock_predictors.py`
  (Sat → weekend). Other states: Mon 08:00 before-open, Mon 12:00 open, Mon
  17:00 after-close, 2026-05-25T10:00 Memorial Day.
- `requirements.txt` is duplicated (merge artifact) — not yet cleaned.
- Stray local branch `feature/improve-grid-layout` exists (pre-existing, leave it).
