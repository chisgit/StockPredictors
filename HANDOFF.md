# Handoff

## Workspace
- Path: `c:\Users\User\StockPredictors`
- Date: 2026-06-17
- Branch: `main` @ `0482643` — working tree clean, pushed
- No active worktrees

---

## Workflow rules (binding on next session)
- **Manual test before commit/PR/merge** — verify each UI change in the live app before staging.
- **Large sections → dedicated plan** — split big sections into own `*_PLAN.md`, defer master row. See §7 → [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md).
- **pytest -q green before merge** — every branch runs the full suite.

---

## App shape / orientation

| Item | Detail |
|------|--------|
| Entry point | [stock_predictors.py](stock_predictors.py) |
| UI render | [render_ui.py](render_ui.py), [render_helpers.py](render_helpers.py) |
| Session state | [session_state.py](session_state.py) |
| Search logic | [render_helpers.py:401](render_helpers.py#L401) `search_and_add_ticker` |
| Plan docs | [UI_UX_PLAN.md](UI_UX_PLAN.md) · [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md) |
| Run (main) | `C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe run stock_predictors.py` (port 8501) |
| Run (worktree) | `-WorkingDirectory <worktree-path>` + port 8502+ |

---

## Done (merged to main) — full history

| § | Change | Ref |
|---|--------|-----|
| §0 | Holiday/weekend-aware `market_status()` | PR #8 |
| §1 | Single status-driven header | PR #10–12 |
| §3 | Grouped stats panel | PR #13 |
| §2 | Two prediction model cards | PR #16 |
| §1f | Before-open data wiring | PR #15 |
| §1f-δ | Inline Δ label | PR #17 |
| §4+§5 | Close-card color + bold deltas | PR #18 |
| §6 | Strip chart chrome | PR #14 |
| DM3 | Real grouping card | `4e7af7b` |
| DM4 | Chart title tile border + card shading | `ec630ad` |
| DM5 | Stats grid 3+3 split | `ca50213` |
| DM6 | Next-day grouping card | `ca50213` |
| Palette | Full dark theme token dict | `ca50213` |
| Render search stabilization | Search rerun loop fix | PR #21 |
| Search autoselect + clear + loading | One-search auto-select, counter key clear, `st.status` loading, empty default, hint text | PR #22 (`b731a9b`) |
| Search edge-case fixes | Guard, yfinance validation, spinner, flash suppression | `b45e817`–`0482643` |
| Card spacing | Uniform today + next-day card spacing | PR #23 (`7afe50b`) |
| Market status titles | Simplified header copy | `89227ef` |

---

## Key domain facts (permanent)

### Streamlit widget reset — counter key pattern
`del st.session_state[widget_key]` and `st.session_state[widget_key] = ""` both fail to clear a text_input on programmatic reruns — Streamlit uses the browser's last value. **The only reliable reset**: increment a counter in session state and use it in `key=`. The browser has no history for the new key, so the widget renders empty.

```python
# session_state: search_input_counter = 0
key = f"new_ticker_input_{st.session_state.search_input_counter}"
new_ticker = st.text_input("Search:", value="", key=key)
# On success:
st.session_state.search_input_counter += 1
st.rerun()
```

### Streamlit multiselect auto-select — rerun required
Multiselect renders **before** `search_and_add_ticker` runs. Appending to `selected_tickers` mid-render has no effect on the current cycle. `st.rerun()` after add is the only way to auto-select in one interaction.

### `search_and_add_ticker` tri-state return
- `True` — ticker added/selected → increment counter + `st.rerun()`
- `None` — guard fired (already processed) → increment counter, no rerun
- `False` — invalid/empty → leave bar so warning stays visible

### `last_processed_ticker_search` guard — known limitation
Set on first processing, never cleared. If user removes a ticker from multiselect and re-searches it, guard fires → `None` → bar clears without re-adding. Fix: clear guard in multiselect `on_change` when ticker is removed. Not yet done — see follow-ups.

---

## Next task — DM1 + DM2 (dark mode polish, final items)

Branch: `feat/dm1-dm2-polish` off `main`  
Worktree: `C:\Users\User\SP-dm1-dm2`

### DM1 — Toggle beside title, kill sidebar
**File:** [render_ui.py](render_ui.py)  
Move 🌙/☀️ toggle out of `st.sidebar` onto the title row. Use `col_icon, col_title = st.columns([1, 12])` — button in `col_icon`, `st.title` in `col_title`. Keep flip logic (`st.session_state.theme` swap + `st.rerun()`) verbatim.  
Accept: no sidebar; toggle on title row; theme still flips.

### DM2 — Subtitle color = `text_delta_label` token
**File:** [display_market_status.py](display_market_status.py)  
Subtitle at line 44 hardcodes `#555555`. Change to `THEME[theme]["text_delta_label"]`. Thread `theme` from `st.session_state` into `generate_market_status_header()` and `display_market_status()`.  
Accept: subtitle color equals `text_delta_label`; no hardcoded hex.

**Steps:**
1. `git worktree add C:\Users\User\SP-dm1-dm2 -b feat/dm1-dm2-polish main`
2. DM2 first (independent, simpler) — one commit
3. DM1 — one commit
4. Extend `tests/test_dark_theme.py` — subtitle carries `text_delta_label` (DM2); toggle on title row not sidebar (DM1)
5. `pytest -q` green
6. Manual test in live app — both themes, verify toggle placement and subtitle color
7. Push → PR → Sr. dev review → merge → delete

---

## Open follow-ups

| Item | Where tracked |
|------|---------------|
| `last_processed_ticker_search` never cleared | Domain facts above — fix: clear in multiselect on_change when ticker removed |
| DM1 toggle beside title | [DARK_MODE_PLAN.md §DM1](DARK_MODE_PLAN.md) |
| DM2 subtitle color token | [DARK_MODE_PLAN.md §DM2](DARK_MODE_PLAN.md) |

---

## Locked decisions
- Counter key (`search_input_counter`) is the canonical Streamlit text_input reset — do not revert to key deletion or assignment.
- `st.rerun()` after ticker add is required for one-cycle multiselect sync.
- `search_and_add_ticker` tri-state — do not collapse to bool; `None` must not force rerun.
- Dark mode default; light toggle via `st.session_state.theme`.

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

# Run tests
Set-Location "C:\Users\User\StockPredictors"; & ".\venv\Scripts\pytest.exe" -q
```

Force market states:
```powershell
$env:MARKET_NOW = "2026-06-17 10:00"   # trading day, open
$env:MARKET_NOW = "2026-06-14 18:00"   # Saturday, closed
$env:MARKET_NOW = ""                    # clear override
```
