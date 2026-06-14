# HANDOFF — session handoff

Updated: 2026-06-14 · Branch: `main` @ `7ec9110` · Working tree: clean (untracked: `.kilo/`, `.opencode/`, `before_predict.png`, `screenshot.png`, `take_screenshot.py`)

Worktree branch: `feat/delta-lowered` @ `b724d3f` · Working tree: dirty (`render_helpers.py` modified — delta label moved below price diff, NOT yet committed)

## What this work is
Executing the UI/UX rework in [UI_UX_PLAN.md](UI_UX_PLAN.md) — fixing the
"visually disturbing" prediction results view (Streamlit app). Sections §0–§7,
done one per branch, dependency-ordered. Read UI_UX_PLAN.md first; its
**Progress** table is the source of truth for what's done.

This session added **§1f-δ-lowered**: moving the inline Δ label from beside
the price difference to below it (still inside the model card).

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
- Tests: `venv/Scripts/python.exe -m pytest tests/ -q` (67 passing on main).

## Done (merged to main)
- **§0** holiday/weekend-aware `market_status()` (PR #8) + `MARKET_NOW` env
  override (PR #9) + gitignore `*_data.csv`.
  Tests [tests/test_market_status.py](tests/test_market_status.py) (18 cases).
- **§1** single status-driven header (PR #10) + style polish (PR #11, #12).
  [display_market_status.py](display_market_status.py) `generate_market_status_header()`.
  Tests [tests/test_header.py](tests/test_header.py) (12 cases).
- **§1f** before-open prediction-vs-actual data wiring (PR #15).
  Tests [tests/test_data_wiring.py](tests/test_data_wiring.py) (9 cases).
- **§1f-δ** inline Δ label on prediction cards (PR #17).
  `_delta_caption(is_open)` → `"Δ from close"` / `"Δ from last traded"`.
  Label inline left of `(+$X.XX)` in each model card. No standalone caption.
- **§3** grouped stats panel (PR #13).
  [render_helpers.py](render_helpers.py) `create_grid_display()`.
  Tests [tests/test_stats_panel.py](tests/test_stats_panel.py).
- **§6** stripped chart chrome (PR #14).
  Tests [tests/test_chart_chrome.py](tests/test_chart_chrome.py) (6 cases).
- **§2** two prediction model cards (PR #16).
  Side-by-side Linear Regression / XGBoost cards with predicted price + colored delta.
  Tests [tests/test_prediction_cards.py](tests/test_prediction_cards.py) (11 cases).
- **§4+§5** close-card neutral bg + color number + equal-case delta (PR #18).
  §4: Close/Last-Traded card bg/border neutral (matches other cards); value text
  colored via `_close_color(close, prev_close)` — green/red/neutral.
  §5: Prediction card deltas explicit 3-way: up=green/`+`, down=red/`-`,
  equal=neutral `#475569`/no sign. Bold (`font-weight: 700`) was already wired
  by PR #17.
  Tests [tests/test_close_color_deltas.py](tests/test_close_color_deltas.py) (9 cases).

## In-progress (uncommitted, worktree)
- **§1f-δ-lowered:** Move Δ label below price diff in card.
  Branch `feat/delta-lowered`, worktree `StockPredictors-delta-lowered`.
  `render_helpers.py:_model_card_html()` — delta label moved from inline flex
  row to standalone div below the `(+$X.XX)` value. **NOT committed** — waiting
  on user visual approval. Launch via port 8502:
  ```powershell
  Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
    -ArgumentList "run","stock_predictors.py","--server.port","8502" `
    -WorkingDirectory "C:\Users\User\StockPredictors-delta-lowered" -WindowStyle Normal
  ```

## Key domain facts
- Close price = last traded price of regular session. `Δ from close` (closed)
  and `Δ from last traded` (open) differ only during market hours.
- Today-target models use today's intraday OHLCV as features
  ([feature_engineering.py](feature_engineering.py)). Before the bell there is
  no today row — "today" prediction re-predicts last session.
- Equal-delta test: pass `[actual_close, actual_close]` as predictions to
  force `delta = 0` and verify neutral rendering.

## Next task — complete §1f-δ-lowered → then §7: dark theme
1. **Finish §1f-δ-lowered:** Launch the worktree app (port 8502) → user
   confirms the delta-label-below layout looks good → commit → test →
   push → PR → merge. Branch: `feat/delta-lowered`.
2. **§7: dark theme + light toggle** — Branch: `feat/dark-theme-toggle` (create off fresh `main`).
   - Default full dark mode for prediction cards and stats panel to match TradingView chart.
   - Palette: bg `#0f172a`/`#111827`, borders `rgba(148,163,184,0.18)`, text `#f8fafc`/`#cbd5e1`.
   - Add theme-toggle icon (🌙/☀️) stored in `st.session_state.theme` (default `"dark"`).
   - Drive colors from a token dict `THEME["dark"]`/`THEME["light"]` consumed by
     `display_predictions`, `create_grid_display`, and the chart builder.
   - Align all cards to chart's left/right edges (same max-width/padding).

## Open follow-ups
- **Large card container refinement.** §2 wrapped each ticker section in a
  unified slate-100 card — visual treatment is placeholder. Tracked in
  UI_UX_PLAN.md §7 scope.

## Decisions locked
- Calendar lib: `pandas_market_calendars` (XNYS), offline.
- User sees only Open/Closed; Closed shows last session date.
- Delta label: `Δ from close` (closed) / `Δ from last traded` (open). Now below
  price diff (in progress).
- Close card: neutral background, color-only-number (not whole-card tint).
- Equal delta: neutral grey `#475569`, no `+`/`-` sign.
- Theme (§7): default full dark + light-toggle icon, via theme token dict.
- Half-days: out of scope (hardcoded 09:30/16:00).

## Workflow / conventions
- One branch per section. Branch off main → implement → **manual test** →
  tests → `pytest -q` green → push → `gh pr create` → `gh pr merge --merge`.
- Commits end with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Each section gets its own isolated test file.

## Gotchas
- Status only renders after clicking **Predict** (runs `yf.download` — live
  network, can lag/flake).
- Manual UI states via `MARKET_NOW`:
  `$env:MARKET_NOW="2026-06-13T08:00"; venv/Scripts/streamlit.exe run stock_predictors.py`
  Other states: `T12:00` open, `T17:00` after-close, `2026-05-25T10:00` Memorial Day.
- **Worktree teardown:** Streamlit holds the worktree directory open. Kill by
  port before `git worktree remove`:
  ```powershell
  netstat -ano | findstr ":8502" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  git worktree remove "C:/Users/User/StockPredictors-<branch>" --force
  git branch -d <branch>
  ```
- Rebased feature branches need `git push --force-with-lease` (not plain push).
- `.opencode/`, `.kilo/` untracked directories — leave them (not part of project).
- **CLAUDE.md symlinks set up this session:**
  - `~/.claude/CLAUDE.md` → global opencode config (three-tier agent routing)
  - `~/StockPredictors` → active repo (`/mnt/c/Users/User/StockPredictors`)
  Both resolve into system prompt as fallback instructions.
