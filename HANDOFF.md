# Handoff

## Workspace
- Path: `c:\Users\User\StockPredictors`
- Date: 2026-06-16
- Branch: `main` @ `b8dc3a3` — working tree clean (except `.claude/settings.json` uncommitted)
- Active fix worktree: `C:\Users\User\SP-search-fix` on `feat/search-autoselect`

---

## Workflow rules (binding on next session)
- **Manual test before commit/PR/merge** — verify each UI change in the live app before staging anything. See [memory/manual-test-before-checkin.md](../../../.claude/projects/c--Users-User-StockPredictors/memory/manual-test-before-checkin.md).
- **Large sections → dedicated plan** — when a plan section grows big, split it into its own `*_PLAN.md` and update the master row. See [UI_UX_PLAN.md](UI_UX_PLAN.md) §7 → [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md).
- **pytest -q green before merge** — every branch runs the full suite.

---

## App shape / orientation

| Item | Detail |
|------|--------|
| Entry point | [stock_predictors.py](stock_predictors.py) |
| UI render | [render_ui.py](render_ui.py), [render_helpers.py](render_helpers.py) |
| Session state | [session_state.py](session_state.py) |
| Trace util | [trace_utils.py](trace_utils.py) |
| Plan docs | [UI_UX_PLAN.md](UI_UX_PLAN.md) · [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md) |
| Run (main) | `C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe run stock_predictors.py` (port 8501) |
| Run (worktree) | Point `-WorkingDirectory` at the worktree path; use port 8502+ |

---

## Done (merged to main)

| § | Change | Ref |
|---|--------|-----|
| §0 | Holiday/weekend-aware `market_status()` | PR #8 |
| §1 | Single status-driven header | PR #10, #11, #12 |
| §3 | Grouped stats panel | PR #13 |
| §2 | Two prediction model cards | PR #16 |
| §1f | Before-open data wiring | PR #15 |
| §1f-δ | Inline Δ label | PR #17 |
| §4+§5 | Close-card number color + bold deltas | PR #18 |
| DM3 | Real grouping card (native `st.container`) | `4e7af7b` |
| DM5 | Stats grid 3+3 split | merged `ca50213` |
| DM6 | Next-day grouping card | merged `ca50213` |
| Palette | Full dark theme token dict | merged `ca50213` |
| Render search stabilization | Ticker search rerun loop fix, input hardening | PR #21 (`b8dc3a3`) |

---

## Key domain facts (this session)

### Search flow — root cause of the two-searches bug
Streamlit renders widgets top-to-bottom, one complete pass per rerun. The multiselect
(`stock_multiselect`) renders **before** `search_and_add_ticker` runs. When search adds
a ticker to `st.session_state.selected_tickers`, the multiselect has already painted
with the old selections. The stale-widget deletion logic in [render_ui.py:227-246](render_ui.py#L227-L246)
DOES exist and WOULD sync on the next render, but that next render only fires when the
user does something else (the second search). With no `st.rerun()` after search succeeds,
the auto-select never fires in the same cycle.

The search bar also never cleared because resetting `st.session_state.new_ticker = ""`
is a different key from the widget key `"new_ticker_input"`. Streamlit uses the widget
key as ground truth, not the backing variable.

### Fix applied in `feat/search-autoselect`
**[render_helpers.py:401](../SP-search-fix/render_helpers.py#L401) — `search_and_add_ticker` now returns:**
- `True` — ticker added or selected → caller clears bar and reruns
- `None` — guard fired (already processed) → caller clears bar, no forced rerun
- `False` — invalid/empty ticker → keep bar so warning is visible

**[render_ui.py:283](../SP-search-fix/render_ui.py#L283) — acts on return value:**
```python
if search_result is True:
    del st.session_state["new_ticker_input"]   # clears widget, not just backing var
    st.session_state.new_ticker = ""
    st.rerun()                                  # forces multiselect to pick up new selections
elif search_result is None:
    del st.session_state["new_ticker_input"]   # clear bar, no forced rerun
    st.session_state.new_ticker = ""
# False: leave bar for user to see warning
```

---

## In-progress work

### `feat/search-autoselect` → `C:\Users\User\SP-search-fix`
**Status: code written, app running on port 8503, NOT yet manually tested or committed.**

**Next concrete steps:**
1. Open http://localhost:8503 in a browser.
2. Type a valid ticker (e.g. `NVDA`) in the search bar and press Enter.
3. Verify: (a) ticker appears selected in the multiselect immediately, (b) search bar clears.
4. Type an invalid ticker (e.g. `ZZZZZ`) → verify warning shows, bar keeps the text.
5. Type the same ticker again after selecting it → verify bar clears, no loop.
6. `pytest -q` green in the worktree.
7. Commit in the worktree, push, open PR against `main`.

**Acceptance criteria:**
- One search → ticker auto-selected in multiselect (no second search needed).
- Search bar clears after valid ticker.
- Invalid ticker: warning stays visible, bar keeps text.
- No rerun loop.

---

### Dark mode — DM1 and DM2 pending
See [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md) for full spec.

**DM4 was the last in-progress item** per the plan header but DM4 also appears
merged (it's in the recent commit history as `feat(§7/dm4)`). Verify against
live app before assuming DM4 is done. DM1 and DM2 are confirmed pending.

| Item | What | File | Status |
|------|------|------|--------|
| DM1 | Toggle beside title, kill sidebar | [render_ui.py](render_ui.py) | ❌ pending |
| DM2 | Subtitle color = `text_delta_label` token | [display_market_status.py](display_market_status.py) | ❌ pending |
| DM4 | Chart title tile — full border + card shading | [render_helpers.py](render_helpers.py) | verify ✅ |

Dark mode branch: check `feat/dark-mode-polish` (may be in worktree at
`C:\Users\User\StockPredictors-dark-mode-worktree`).

---

## Open follow-ups

| Item | Where tracked |
|------|---------------|
| DM1 — toggle beside title | [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md) §DM1 |
| DM2 — subtitle color token | [DARK_MODE_PLAN.md](DARK_MODE_PLAN.md) §DM2 |
| `last_processed_ticker_search` not cleared on removal | Known bug — if user removes a ticker from multiselect then re-searches it, guard fires and returns None; bar clears but ticker isn't re-added. Future fix: clear guard when multiselect removes the ticker. |

---

## Locked decisions

- `search_and_add_ticker` returns a tri-state (`True`/`None`/`False`) — do not collapse to bool; the `None` case (guard fired) must NOT force a rerun.
- Widget key to delete on clear: `"new_ticker_input"` (the `key=` on `st.text_input`).
- `st.rerun()` is the only reliable way to force a multiselect to re-render with new `default`; the stale-widget-deletion approach (lines 227-246) works but only when something else triggers a rerun first.

---

## How to verify / run

```powershell
# Run main app
& "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" run stock_predictors.py

# Run search-fix worktree (port 8503)
Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
  -ArgumentList "run","stock_predictors.py","--server.port","8503" `
  -WorkingDirectory "C:\Users\User\SP-search-fix" `
  -WindowStyle Normal

# Kill port 8503
netstat -ano | findstr ":8503" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

# Run tests
Set-Location "C:\Users\User\StockPredictors"; & ".\venv\Scripts\pytest.exe" -q
```

Force market states via env var (§0):
```powershell
$env:MARKET_NOW = "2026-06-15 10:00"   # trading day, open
$env:MARKET_NOW = "2026-06-14 18:00"   # Saturday, closed
$env:MARKET_NOW = ""                    # clear override
```
