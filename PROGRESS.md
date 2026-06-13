# PROGRESS — session handoff

Updated: 2026-06-13 · Branch: `main` @ `76a34cd` · Working tree: clean

## What this work is
Executing the UI/UX rework in [UI_UX_PLAN.md](UI_UX_PLAN.md) — fixing the
"visually disturbing" prediction results view (Streamlit app). Sections §0–§7,
done one per branch, dependency-ordered. Read UI_UX_PLAN.md first; its
**Progress** table is the source of truth for what's done.

## App shape (orientation)
- Streamlit app. Entry: [stock_predictors.py](stock_predictors.py)
  (`streamlit run stock_predictors.py`).
- Results UI: [render_ui.py](render_ui.py) `display_results()` →
  [render_helpers.py](render_helpers.py) (prediction pill, TradingView chart,
  stats grid) + [display_market_status.py](display_market_status.py) (header).
- Market status: [utils.py](utils.py) `market_status()` / `is_trading_day()`.
- venv: `./venv/Scripts/python.exe`, `./venv/Scripts/streamlit`.

## Done (merged to main)
- **§0** holiday/weekend-aware `market_status()` (PR #8). Non-trading day →
  `AFTER_MARKET_CLOSE`. Uses `pandas_market_calendars` (XNYS), `is_trading_day`
  cached per date. Same 3-string return — callers untouched.
- **Tests** [tests/test_market_status.py](tests/test_market_status.py) — 18
  cases, time injected via `monkeypatch` of `utils.get_nyse_datetime`. Run:
  `./venv/Scripts/python.exe -m pytest tests/ -q`.
- **`MARKET_NOW` env override** (PR #9) — force any instant for manual UI QA:
  `$env:MARKET_NOW="2026-06-13T12:00"; ./venv/Scripts/streamlit run stock_predictors.py`
  (Sat noon → Closed; the §0 bug case). Unset in prod. Unit tests don't use it.
- **gitignore** `*_data.csv` (untracked AAPL/MSFT/NVDA/TSLA) — they kept
  dirtying the tree on git ops. Files stay on disk.

## Next: §1 — single status-driven header
Branch `feat/single-header` off fresh `main`. Plan §1 in UI_UX_PLAN.md. Summary:
- Collapse the two competing headers (centered date `<h2>` in
  display_market_status.py + left `st.subheader("Today's Close Predictions")`
  at [render_ui.py:58](render_ui.py#L58)) into ONE left-aligned status-driven
  header with inline icon.
- Subtitle binary on `market_status()`: **Open** → no subtitle; **Closed**
  (before/after/weekend/holiday) → show last session date from
  `data.index[-1].date()`. Remove hardcoded "Today's closing prices are final."
- Add `tests/test_header.py` (smoke-test the rendered HTML strings per status).

## Decisions locked (see UI_UX_PLAN.md "Resolved Decisions")
- Calendar lib: `pandas_market_calendars` (XNYS), offline.
- User sees only Open/Closed; Closed shows last session date.
- Theme (§7): default full dark + light-toggle icon, via a theme token dict.
- Half-days: out of scope (hardcoded 09:30/16:00).

## Workflow / conventions
- One branch per section: branch off main → implement → tests → `pytest -q`
  green (merge gate) → push → `gh pr create` → `gh pr merge --merge
  --delete-branch`.
- ⚠️ Dirty `*_data.csv` previously broke `gh pr merge --delete-branch`
  (post-merge pull blocked). Now gitignored — should be fine.
- Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Each section gets its own isolated test file so a change in one branch that
  breaks another is caught by the full suite.

## Gotchas
- Status only renders after clicking **Predict** (runs `yf.download` — live
  network, can lag/flake).
- `requirements.txt` is duplicated (merge artifact) — not yet cleaned; add deps
  once, pip dedupes.
- Stray local branch `feature/improve-grid-layout` exists (pre-existing, not
  part of this work — leave it).
