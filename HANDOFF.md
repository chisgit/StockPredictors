# HANDOFF — session handoff

Updated: 2026-06-13 · Branch: `main` @ `9137123` · Working tree: dirty (untracked: `.opencode/`, `before_predict.png`, `screenshot.png`, `take_screenshot.py`)

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

## ⚠️ Worktree setup (use these — do NOT work in main repo on feature branches)
Each active branch has its own isolated working directory.

| Branch | Worktree path |
|---|---|
| `main` | `c:\Users\User\StockPredictors` |
| `feat/prediction-cards` | `c:\Users\User\StockPredictors-prediction-cards` |

**venv is only in the main repo.** From any worktree, use absolute path:
`c:\Users\User\StockPredictors\venv\Scripts\python.exe`
`c:\Users\User\StockPredictors\venv\Scripts\streamlit.exe`

To add a new worktree for a new branch:
```powershell
git worktree add "..\StockPredictors-<branch-slug>" <branch-name>
```

## App shape (orientation)
- Streamlit app. Entry: [stock_predictors.py](stock_predictors.py)
  (`c:\Users\User\StockPredictors\venv\Scripts\streamlit.exe run stock_predictors.py`
  — run from the worktree dir for that branch).
- Results UI: [render_ui.py](render_ui.py) `display_results()` →
  [render_helpers.py](render_helpers.py) (prediction cards, TradingView chart,
  stats grid) + [display_market_status.py](display_market_status.py) (header).
- Market status: [utils.py](utils.py) `market_status()` / `is_trading_day()` /
  `next_trading_day()`.
- Tests: `<venv>/python.exe -m pytest tests/ -q` (57 passing on main).

## Done (merged to main)
- **§0** holiday/weekend-aware `market_status()` (PR #8) + `MARKET_NOW` env
  override (PR #9) + gitignore `*_data.csv`.
  Tests [tests/test_market_status.py](tests/test_market_status.py) (18 cases).
- **§1** single status-driven header (PR #10) + style polish (PR #11, #12).
  [display_market_status.py](display_market_status.py) `generate_market_status_header()`:
  - `BEFORE_MARKET_OPEN` → 🔴 "Market Closed" + "Displaying Predictions for `<last session>`"
  - `MARKET_OPEN` → 🔔 "Live — Last Traded" (no subtitle)
  - `AFTER_MARKET_CLOSE` trading day → 🔴 "Today's Close Predictions and Actuals"
  - `AFTER_MARKET_CLOSE` weekend/holiday → 🔴 "Market Closed Weekend/Holiday" + "Displaying Predictions for `<last session>`"
  - Dated next-day header via `generate_next_day_header()`.
  - Tests [tests/test_header.py](tests/test_header.py) (12 cases).
- **§3** grouped stats panel (PR #13).
  [render_helpers.py](render_helpers.py) `create_grid_display()` now accepts
  `session_date`. Wraps OHLC/Volume cards in bordered panel; header = "Today's
  Data" (open) or last session date e.g. "Friday, June 12" (closed).
  [render_ui.py](render_ui.py) passes `last_available_date` as `session_date`.
  Tests [tests/test_stats_panel.py](tests/test_stats_panel.py) (10 cases).
- **§6** stripped chart chrome (PR #14).
  Removed "TradingView Style Chart" eyebrow + "OHLC candles" caption.
  Extracted `generate_chart_widget_html(ticker, chart_json)` — pure fn.
  Tests [tests/test_chart_chrome.py](tests/test_chart_chrome.py) (6 cases).
- **§1f** before-open prediction-vs-actual data wiring (PR #15).
  `display_results()` now resolves `market_status()` once; when not open,
  shows caption under each prediction strip: "Δ vs Friday, Jun 12 actual close · $X".
  Pure helper `_session_ref_caption(session_date, actual_close)` in `render_ui.py`.
  Tests [tests/test_data_wiring.py](tests/test_data_wiring.py) (11 cases).

## Key domain fact
Today-target models use **today's intraday OHLCV** as features
([feature_engineering.py:179](feature_engineering.py#L179)). **Before the bell
there is no today row** — "today" prediction is a re-prediction of the last
session. Real forward prediction is the next-day model. This is why before open
the main section is framed as a past-session accuracy recap.

## Active branches

### §2 — two prediction model cards (`feat/prediction-cards`)
Worktree: `c:\Users\User\StockPredictors-prediction-cards`
Implementation complete, **awaiting manual test sign-off + commit**.
- `render_helpers.py` — `_model_card_html`, `_prediction_cards_container_html`,
  `generate_prediction_cards_html`, updated `display_predictions` (3-arg).
  Replaces `preds_sameline()`.
- `render_ui.py` — removed `preds_sameline` import; calls
  `display_predictions(ticker, predictions, current_val)` directly.
- `tests/test_prediction_cards.py` (10 cases) — all passing.
- **Next:** launch app from worktree (`<venv>/streamlit.exe run stock_predictors.py`),
  verify two side-by-side model cards render, get user "looks good", then
  commit, push, PR, merge.

## Concurrency map — remaining work

```
main (§0✅ §1✅ §1f✅ §3✅ §6✅)
│
└── §2  feat/prediction-cards  render_helpers.py:61-90  🔄 IN PROGRESS (worktree ready)
        │
        └── §4+§5  feat/close-color-deltas  (after §2 merges)
                │
                └── §7  feat/dark-theme-toggle  (last — after §2 §4+§5)
```

### Sequential gates
- **§4+§5 after §2** — §5 applies to the new cards §2 creates.
- **§7 last** — final theme pass over completed markup.

## Decisions locked (see UI_UX_PLAN.md "Resolved Decisions")
- Calendar lib: `pandas_market_calendars` (XNYS), offline.
- User sees only Open/Closed; Closed shows last session date.
- "Displaying" subtitle cue only on closed-with-past-session states.
- Theme (§7): default full dark + light-toggle icon, via a theme token dict.
- Half-days: out of scope (hardcoded 09:30/16:00).

## Agent coordination (startflow/handoff skills)
- `/startflow` claims next unclaimed task: reads concurrency map, checks
  `git branch -a` for existing remote branches, creates + pushes branch (atomic
  lock), marks plan doc `🔄 in progress` on feature branch.
- Use dedicated worktree for each feature branch — see table above.
- `/handoff` writes this file + commits on current branch.

## Workflow / conventions
- One branch per section + one worktree per branch.
- branch off main → implement in worktree → **manual test** →
  tests → `pytest -q` green → push → `gh pr create` → `gh pr merge --merge
  --delete-branch` → `git worktree remove`.
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
