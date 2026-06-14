# HANDOFF — session handoff

Updated: 2026-06-13 · Branch: `main` @ `d47efb3` · Working tree: clean (untracked: `.opencode/`, `before_predict.png`, `screenshot.png`, `take_screenshot.py`)

## What this work is
Executing the UI/UX rework in [UI_UX_PLAN.md](UI_UX_PLAN.md) — fixing the
"visually disturbing" prediction results view (Streamlit app). Sections §0–§7,
done one per branch, dependency-ordered. Read UI_UX_PLAN.md first; its
**Progress** table is the source of truth for what's done.

## ⚠️ Workflow rule (honor every session)
**Manually test each feature/fix in the running app BEFORE commit/PR/merge.**
After implementing, launch the app, give the user the localhost URL + the
expected render, and **wait for their "looks good"** before `git commit` / PR /
merge. Don't batch commit+PR+merge until they've eyeballed it.

## App shape (orientation)
- Streamlit app. Entry: [stock_predictors.py](stock_predictors.py)
  (`c:\Users\User\StockPredictors\venv\Scripts\streamlit.exe run stock_predictors.py`).
- Results UI: [render_ui.py](render_ui.py) `display_results()` →
  [render_helpers.py](render_helpers.py) (prediction cards, TradingView chart,
  stats grid) + [display_market_status.py](display_market_status.py) (header).
- Market status: [utils.py](utils.py) `market_status()` / `is_trading_day()` /
  `next_trading_day()`.
- Tests: `<venv>/python.exe -m pytest tests/ -q` (61 passing on main).

## Done (merged to main)
- **§0** holiday/weekend-aware `market_status()` (PR #8) + `MARKET_NOW` env
  override (PR #9) + gitignore `*_data.csv`.
  Tests [tests/test_market_status.py](tests/test_market_status.py) (18 cases).
- **§1** single status-driven header (PR #10) + style polish (PR #11, #12).
  [display_market_status.py](display_market_status.py) `generate_market_status_header()`.
  Tests [tests/test_header.py](tests/test_header.py) (12 cases).
- **§1f** before-open prediction-vs-actual data wiring (PR #15).
  Caption under predictions when market closed: "Δ vs session actual close".
  Tests [tests/test_data_wiring.py](tests/test_data_wiring.py) (11 cases).
- **§3** grouped stats panel (PR #13).
  [render_helpers.py](render_helpers.py) `create_grid_display()` — grouped
  OHLC/Volume cards. Tests [tests/test_stats_panel.py](tests/test_stats_panel.py).
- **§6** stripped chart chrome (PR #14). Removed eyebrow + caption labels.
  Tests [tests/test_chart_chrome.py](tests/test_chart_chrome.py) (6 cases).
- **§2** two prediction model cards (PR #16).
  Replaced cramped `preds_sameline()` with two side-by-side model cards
  (Linear Regression, XGBoost). Each card: model name, predicted price,
  colored delta. Ticker moved to section header with accent bar. Chart
  and stats outer borders stripped, wrapped in unified ticker container.
  Tests [tests/test_prediction_cards.py](tests/test_prediction_cards.py)
  (10 cases, updated for 2-arg signature).

## Key domain fact
Today-target models use **today's intraday OHLCV** as features
([feature_engineering.py:179](feature_engineering.py#L179)). **Before the bell
there is no today row** — "today" prediction is a re-prediction of the last
session. Real forward prediction is the next-day model.

## Next task — §4+§5: close-card color + bold deltas
Branch: `feat/close-color-deltas` (create off fresh `main`).
- §4: Close/Last-Traded card — neutral background, color only the number
  (green up, red down, neutral equal). No whole-card tint.
- §5: Deltas in prediction cards — bold (`font-weight: 700`), slightly larger.
  Already partially in place from §2; verify and lock with a test.

## Open follow-ups
- **Large card container refinement.** §2 wrapped each ticker section in a
  unified card (slate-100 background). The visual treatment is placeholder —
  needs polish. Tracked in UI_UX_PLAN.md as part of §7 scope or separate.

## Decisions locked
- Calendar lib: `pandas_market_calendars` (XNYS), offline.
- User sees only Open/Closed; Closed shows last session date.
- Theme (§7): default full dark + light-toggle icon, via a theme token dict.
- Half-days: out of scope (hardcoded 09:30/16:00).

## Workflow / conventions
- One branch per section. branch off main → implement → **manual test** →
  tests → `pytest -q` green → push → `gh pr create` → `gh pr merge --merge
  --delete-branch`.
- Commits end with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Each section gets its own isolated test file.

## Gotchas
- Status only renders after clicking **Predict** (runs `yf.download` — live
  network, can lag/flake).
- Manual UI states via `MARKET_NOW`:
  `$env:MARKET_NOW="2026-06-13T08:00"; <venv>/streamlit.exe run stock_predictors.py`
  Other states: `T12:00` open, `T17:00` after-close, `2026-05-25T10:00` Memorial Day.
- `.opencode/` untracked directory — leave it (not part of this project).
- `requirements.txt` is duplicated (merge artifact) — not yet cleaned.
- Stray local branch `feature/improve-grid-layout` exists (pre-existing, leave it).
