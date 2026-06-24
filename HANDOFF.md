# HANDOFF — StockPredictors + Carousel Project
Updated: 2026-06-24 | Branch: `main` @ `42db1dc` (cache fix merged) | Carousel: complete

## Workspace
- Path: `c:\Users\User\StockPredictors`
- Production: `https://stockpredictors.onrender.com` (Render `srv-csvanmqj1k6c73c3dnt0`) — LIVE, verified working (GOOGL predictions + chart render)
- Carousel: `C:\Users\User\StockPredictors-carousel\` — 7-slide LinkedIn deck on agentic loops
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

---

## LinkedIn Carousel Project — NEAR FINAL

**Status:** All 7 slides rendered + PDF assembled (v2), ready to finalize + caption queued.

**Files:** `C:\Users\User\StockPredictors-carousel\`
- HTML: `slide1.html` – `slide7.html` (with embedded CSS: browser-mockup, trace-table, anatomy-steps, before/after, recap-cards)
- PNG: `slide1.png` – `slide7.png` (all 2160×2700, retina 2×)
- PDF: `StockPredictors-agentic-loop-carousel-v2.pdf` (current; original locked)

**Slide layout:**
1. Hook + app mockup (`stockpredictors.onrender.com` address bar + LIVE badge + real GOOGL $346.42/$356.86 + candlestick chart)
2. Voice memo (verbatim, trimmed rate-limit line only)
3. JSON loop spec (juicy bits)
4. Anatomy — six moves (OBSERVE/FIX/GATE/SHIP/DEPLOY/VERIFY)
5. **Verify trace** — the intelligence (DECIDE → OPEN → TYPE → CLICK → CHECK → SANITY → RECONCILE, with judgment calls & green checkmarks)
6. Before/after proof (prod logs showing KeyError 'Date' crash → clean skipped_tickers=[], live predictions)
7. Takeaway (the four pillars: goal/loop/guardrails/feedback)

**Pending cleanup (next session or before upload):**
1. Close original `...-carousel.pdf` file (locked on Windows)
2. Delete `...-carousel-v2.pdf`, rename `slide1-7.png` files if needed, recreate final PDF with original filename
3. Close 2 browser tabs (both `https://stockpredictors.onrender.com/`)

**LinkedIn caption:** Streamlined version (approved) — NO "things become names" paragraph (that's a separate post). Hooks: broken-app story → build-with-AI motivation → DM invite → "short small pieces of value."

**Asset notes:**
- Slide 1 app mockup: CSS-built recreation (not raw screenshot) with real live values from 2026-06-23 run
- All copy matches project voice; theme matches dark-mode Streamlit app
- PDF built via Pillow (PIL); render process uses headless Chrome 2× DPI
