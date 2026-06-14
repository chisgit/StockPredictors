# HANDOFF — session handoff

Updated: 2026-06-13 · Branch: `feat/prediction-cards` @ `c0ce1f7` · Working tree: clean

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
  (`./venv/Scripts/streamlit run stock_predictors.py`).
- Results UI: [render_ui.py](render_ui.py) `display_results()` →
  [render_helpers.py](render_helpers.py) (prediction pill, TradingView chart,
  stats grid) + [display_market_status.py](display_market_status.py) (header).
- Market status: [utils.py](utils.py) `market_status()` / `is_trading_day()` /
  `next_trading_day()`.
- venv: `./venv/Scripts/python.exe`, `./venv/Scripts/streamlit`.
- Tests: `./venv/Scripts/python.exe -m pytest tests/ -q` (40 passing).

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
  Data" (open) or last session date e.g. "Friday, June 12" (closed). Same date
  source as §1 subtitle. [render_ui.py](render_ui.py) passes `last_available_date`
  as `session_date`. Tests [tests/test_stats_panel.py](tests/test_stats_panel.py)
  (10 cases).

## Key domain fact
Today-target models use **today's intraday OHLCV** as features
([feature_engineering.py:179](feature_engineering.py#L179)). **Before the bell
there is no today row** — "today" prediction is a re-prediction of the last
session. Real forward prediction is the next-day model. This is why before open
the main section is framed as a past-session accuracy recap.

## Active: §2 — two prediction model cards
Branch `feat/prediction-cards` (already pushed — claimed). Plan §2 in
UI_UX_PLAN.md. Summary:
- Split `preds_sameline()` ([render_helpers.py:61-77](render_helpers.py#L61-L77))
  into **two side-by-side model cards** (Linear Regression | XGBoost).
- Each card: model name, predicted price (large), delta vs reference price.
- Reference price = Close (after close), Last Traded (open), prev close
  (before open) — same source current strip uses.
- Stack on narrow viewports: CSS grid `auto-fit`, same pattern as stats grid
  ([render_helpers.py:231-232](render_helpers.py#L231-L232)).
- Add `tests/test_prediction_cards.py` (smoke-test HTML per status + delta
  sign).
- **§4+§5 depends on §2** — §5 bolds deltas on the new cards, not the old
  strip. Don't touch delta styling here; leave it for §4+§5.

## Concurrency map — remaining work

```
main (§0✅ §1✅ §3✅)
│
├── §2  feat/prediction-cards  render_helpers.py:61-90   🔄 IN PROGRESS
├── §6  feat/chart-chrome      render_helpers.py:121,124  unclaimed
└── data-wiring (render_ui.py only)                       unclaimed
        │
        └── §4+§5  feat/close-color-deltas  (after §2 merges)
                │
                └── §7  feat/dark-theme-toggle  (last — after §2 §4+§5 §6)
```

### Parallel-safe (non-overlapping files/lines off same main)
- **§6** + **data-wiring** — both unclaimed, safe to start alongside §2.
- §2 touches `render_helpers.py:61-90`; §6 touches lines 121,124; data-wiring
  touches `render_ui.py` only. No conflicts.

### Sequential gates
- **§4+§5 after §2** — §5 applies to the new cards §2 creates.
- **§7 last** — final theme pass over completed markup.

## ⚠️ Open follow-up (own branch — user's call)
**Before-open prediction-vs-actual data wiring.** §1 fixed header labels only.
To make "Displaying Predictions for `<Friday>`" pair Friday's prediction vs
Friday's actual close, change [render_ui.py](render_ui.py) `display_results()`
to feed last-session prediction + actual to main section when
`market_status() != MARKET_OPEN`. See UI_UX_PLAN.md §1 "Follow-up".

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
- `/handoff` writes this file + commits on current branch.
- Skills are global (`~/.claude/skills/`); all file I/O resolves to project root.

## Workflow / conventions
- One branch per section: branch off main → implement → **manual test** →
  tests → `pytest -q` green → push → `gh pr create` → `gh pr merge --merge
  --delete-branch`.
- Commits end with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Each section gets its own isolated test file.
- Docs/HANDOFF updates commit on current branch; plan doc updates on main after merge.

## Gotchas
- Status only renders after clicking **Predict** (runs `yf.download` — live
  network, can lag/flake).
- Manual UI states via `MARKET_NOW`:
  `$env:MARKET_NOW="2026-06-13T08:00"; ./venv/Scripts/streamlit run stock_predictors.py`
  Other states: `T12:00` open, `T17:00` after-close, `2026-05-25T10:00` Memorial Day.
- `.opencode/` untracked directory — leave it (not part of this project).
- `requirements.txt` is duplicated (merge artifact) — not yet cleaned.
- Stray local branch `feature/improve-grid-layout` exists (pre-existing, leave it).
