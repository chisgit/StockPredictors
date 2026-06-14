# HANDOFF — session handoff

Updated: 2026-06-14 · Branch: `main` @ `8555e77` (rebased over PR #17) · Working tree: clean (untracked: `.opencode/`, `before_predict.png`, `screenshot.png`, `take_screenshot.py`, `.kilo/`)

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
- Streamlit app. Entry: [stock_predictors.py](stock_predictors.py).
- **Venv:** `C:/Users/User/StockPredictors/venv/Scripts/streamlit.exe` (project root).
  Worktrees do NOT have their own venv — always launch via main project venv:
  ```powershell
  Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
    -ArgumentList "run","stock_predictors.py","--server.port","8501" `
    -WorkingDirectory "C:\Users\User\StockPredictors" -WindowStyle Normal
  ```
  For worktrees change `-WorkingDirectory` and use port `8502+`.
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
  Tests [tests/test_data_wiring.py](tests/test_data_wiring.py) (11 cases).
- **§1f-δ** inline Δ label on prediction cards (PR #17).
  Replaced `st.caption` with inline label left of `(+$X.XX)` in each model card.
  Closed → `Δ from close`. Open → `Δ from last traded`. Shows in both states.
  `_session_ref_caption` removed; `_delta_caption(is_open)` in [render_ui.py](render_ui.py).
  Label threaded: `display_predictions` → `generate_prediction_cards_html` → `_model_card_html`.
- **§3** grouped stats panel (PR #13).
  [render_helpers.py](render_helpers.py) `create_grid_display()`.
  Tests [tests/test_stats_panel.py](tests/test_stats_panel.py).
- **§6** stripped chart chrome (PR #14).
  Tests [tests/test_chart_chrome.py](tests/test_chart_chrome.py) (6 cases).
- **§2** two prediction model cards (PR #16).
  Replaced `preds_sameline()` with side-by-side Linear Regression / XGBoost cards.
  Tests [tests/test_prediction_cards.py](tests/test_prediction_cards.py) (10 cases).

## Key domain facts
- Close price = last traded price of regular session. `Δ from close` and
  `Δ from last traded` are the same value after the bell; they differ only
  during market hours (live intraday vs prior-day official close).
- Today-target models use **today's intraday OHLCV** as features
  ([feature_engineering.py:179](feature_engineering.py#L179)). Before the bell
  there is no today row — "today" prediction is a re-prediction of the last session.

## Next task — §4+§5: close-card color + bold deltas
Branch: `feat/close-color-deltas` (exists, 1 commit ahead of pre-rebase main; WIP stashed).
- **§4:** Close/Last-Traded card — neutral background, color only the number
  (green up, red down, neutral equal). No whole-card tint.
- **§5:** Deltas in prediction cards — bold (`font-weight: 700`), slightly larger.
  Already partially wired from §2; verify and lock with a test.

**To resume:** `git checkout feat/close-color-deltas && git rebase main && git stash pop`

## Open follow-ups
- **Large card container refinement.** §2 wrapped each ticker section in a
  unified card (slate-100 background). Visual treatment is placeholder — needs
  polish. Tracked in UI_UX_PLAN.md §7 scope or separate.

## Decisions locked
- Calendar lib: `pandas_market_calendars` (XNYS), offline.
- User sees only Open/Closed; Closed shows last session date.
- Theme (§7): default full dark + light-toggle icon, via a theme token dict.
- Half-days: out of scope (hardcoded 09:30/16:00).
- Delta label: `Δ from close` (closed) / `Δ from last traded` (open). Inline
  left of price diff, no price value shown (it's redundant — grid below has it).

## Workflow / conventions
- One branch per section. Branch off main → implement → **manual test** →
  tests → `pytest -q` green → push → `gh pr create` → `gh pr merge --merge`.
- Commits end with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Each section gets its own isolated test file.

## Gotchas
- Status only renders after clicking **Predict** (runs `yf.download` — live
  network, can lag/flake).
- Manual UI states via `MARKET_NOW`:
  `$env:MARKET_NOW="2026-06-13T08:00"; <venv>/streamlit.exe run stock_predictors.py`
  Other states: `T12:00` open, `T17:00` after-close, `2026-05-25T10:00` Memorial Day.
- **Worktree teardown:** Streamlit holds the worktree directory open. Kill the
  process by port before `git worktree remove`:
  ```powershell
  netstat -ano | findstr ":8502" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  git worktree remove "C:/Users/User/StockPredictors-<branch>" --force
  git branch -d <branch>
  ```
- `.opencode/`, `.kilo/` untracked directories — leave them (not part of project).
- `requirements.txt` is duplicated (merge artifact) — not yet cleaned.
- Stray local branch `feature/improve-grid-layout` exists (pre-existing, leave it).
- `CLAUDE.md` added to project root this session — venv path + worktree teardown recipe.
