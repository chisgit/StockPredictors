# HANDOFF — StockPredictors
Updated: 2026-06-17 | Branch: `main` @ `3404a86`

## Workspace
- Path: `c:\Users\User\StockPredictors`
- Production: `https://stockpredictors.onrender.com` (Render `srv-csvanmqj1k6c73c3dnt0`)
- Secondary worktree: none active

## Run commands
```powershell
# Main app
& "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" run stock_predictors.py

# Worktree app (port 8502+)
Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
  -ArgumentList "run","stock_predictors.py","--server.port","8502" `
  -WorkingDirectory "C:\Users\User\StockPredictors-<branch>" -WindowStyle Normal

# Kill a port
netstat -ano | findstr ":<port>" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

# Tests
Set-Location "C:\Users\User\StockPredictors"; & ".\venv\Scripts\pytest.exe" -q

# Force market state
$env:MARKET_NOW = "2026-06-17 10:00"   # open
$env:MARKET_NOW = "2026-06-14 18:00"   # Saturday closed
$env:MARKET_NOW = ""                    # clear
```

## Key files
| Role | File |
|------|------|
| Entry | [stock_predictors.py](stock_predictors.py) |
| UI render | [render_ui.py](render_ui.py), [render_helpers.py](render_helpers.py) |
| Session state | [session_state.py](session_state.py) |
| Data fetch | [data_handler.py](data_handler.py) `fetch_data()` L217 |
| Model train | [model.py](model.py) `train_model()` L10 |

## Active branches
| Branch | Plan doc | Status | Priority |
|--------|----------|--------|----------|
| training-cache | [CACHING_PLAN.md](CACHING_PLAN.md) | ⬜ not started | 1 — next |

## Workflow rules
- Manual test before commit/PR/merge — verify each UI change in the live app
- pytest -q green before merge
- Branch → PR → review → merge → delete (no direct commits to main)
- New feature scope → dedicated `*_PLAN.md`

## Locked decisions
- Streamlit text_input reset: increment `search_input_counter` in session state, use in `key=`
- `st.rerun()` after ticker add — required for multiselect one-cycle sync
- `search_and_add_ticker` tri-state: `True` = added, `None` = guard fired (no rerun), `False` = invalid
- Dark mode default; toggle via `st.session_state.theme`
- Do not validate ticker with yfinance before selecting — prediction pipeline is source of truth
- Use `pandas_market_calendars` / XNYS for trading-day checks, not clock time
