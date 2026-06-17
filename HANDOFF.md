# Handoff

## Workspace
- Path: `c:\Users\User\StockPredictors`
- Date: 2026-06-17
- Branch: `main` @ `ad54c28` — clean, pushed
- No active worktrees

---

## Workflow rules (binding on next session)
- **Manual test before commit/PR/merge** — verify each UI change in the live app before staging.
- **New plan sections → dedicated `*_PLAN.md`** — `DARK_MODE_PLAN.md` archived at `archive/plans/`. Create a new plan file when the next feature scope grows.
- **pytest -q green before merge** — every branch runs the full suite.

---

## App shape / orientation

| Item | Detail |
|------|--------|
| Entry point | [stock_predictors.py](stock_predictors.py) |
| UI render | [render_ui.py](render_ui.py), [render_helpers.py](render_helpers.py) |
| Session state | [session_state.py](session_state.py) |
| Search logic | [render_helpers.py](render_helpers.py) `search_and_add_ticker` |
| Plan doc | [UI_UX_PLAN.md](UI_UX_PLAN.md) — **status: Complete** |
| Run (main) | `C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe run stock_predictors.py` (port 8501) |
| Run (worktree) | `-WorkingDirectory <worktree-path>` + port 8502+ |

---

## Current state — all planned work done

[UI_UX_PLAN.md](UI_UX_PLAN.md) is **Complete**. All §0–§7 visual/UX items, all dark mode items (DM1–DM6), search flow, card spacing, and production search regression — all merged and verified in production. No in-flight branches or worktrees.

---

## Key domain facts (permanent)

### Streamlit widget reset — counter key pattern
`del`/assignment both fail to clear a text_input on programmatic reruns — browser value wins. Only reliable reset: increment `search_input_counter` in session state, use it in `key=`.

### Streamlit multiselect auto-select
Multiselect renders before `search_and_add_ticker`. `st.rerun()` after add is required for one-cycle sync.

### `search_and_add_ticker` tri-state return
- `True` — added/selected → increment counter + `st.rerun()`
- `None` — guard fired → increment counter, no rerun
- `False` — invalid → leave bar, warning visible

---

## Next task

No tracked tasks remain. Next session should identify new feature work and create a new plan doc or add rows to [UI_UX_PLAN.md](UI_UX_PLAN.md).

Possible directions to discuss with user:
- Prediction model improvements (accuracy, new models)
- More tickers / data sources
- Portfolio view / multi-ticker comparison
- Deployment / production hardening

---

## Locked decisions
- Counter key (`search_input_counter`) — canonical Streamlit text_input reset.
- `st.rerun()` after ticker add — required for multiselect sync.
- `search_and_add_ticker` tri-state — `None` must not force rerun.
- Dark mode default; toggle via `st.session_state.theme`.

---

## How to verify / run

```powershell
# Run main app
& "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" run stock_predictors.py

# Run worktree app
Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
  -ArgumentList "run","stock_predictors.py","--server.port","8502" `
  -WorkingDirectory "C:\Users\User\<worktree-dir>" -WindowStyle Normal

# Kill a port
netstat -ano | findstr ":<port>" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

# Tests
Set-Location "C:\Users\User\StockPredictors"; & ".\venv\Scripts\pytest.exe" -q

# Force market state
$env:MARKET_NOW = "2026-06-17 10:00"   # open
$env:MARKET_NOW = "2026-06-14 18:00"   # Saturday closed
$env:MARKET_NOW = ""                    # clear
```
