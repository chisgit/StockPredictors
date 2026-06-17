# StockPredictors Project Notes

This is the shared project-memory file. Keep durable lessons here; use
`HANDOFF.md` only for short-lived session state.

## Running The App

Streamlit venv lives in the main project root:

```powershell
C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe
```

Worktrees do not have their own venv. Always launch with the main venv and set
the working directory to the worktree:

```powershell
Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
  -ArgumentList "run","stock_predictors.py","--server.port","8502" `
  -WorkingDirectory "C:\Users\User\StockPredictors-<branch-name>" `
  -WindowStyle Normal
```

Default ports:

- `8501` for the main worktree
- `8502+` for feature worktrees

## Worktrees

Streamlit holds the worktree directory open. `git worktree remove` can fail with
`Permission denied` if Streamlit is still running against that directory. Kill
the process first:

```powershell
netstat -ano | findstr ":8502" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
```

Then remove the worktree and branch:

```powershell
git worktree remove "C:\Users\User\StockPredictors-<branch-name>" --force
git branch -d <branch-name>
```

Current known secondary worktree:

```text
C:\Users\User\StockPredictors-market-status
branch: investigate-market-status
```

## Production / Render

Production URL:

```text
https://stockpredictors.onrender.com
```

Render service:

```text
Name: StockPredictors
ID: srv-csvanmqj1k6c73c3dnt0
```

Local Render CLI was installed manually because the shell install script needed
`unzip`:

```powershell
C:\Users\User\.local\bin\render-cli\cli_v2.20.0.exe
```

Useful commands:

```powershell
& "$env:USERPROFILE\.local\bin\render-cli\cli_v2.20.0.exe" deploys list srv-csvanmqj1k6c73c3dnt0 --output json
& "$env:USERPROFILE\.local\bin\render-cli\cli_v2.20.0.exe" deploys create srv-csvanmqj1k6c73c3dnt0 --confirm --output json
& "$env:USERPROFILE\.local\bin\render-cli\cli_v2.20.0.exe" logs --resources srv-csvanmqj1k6c73c3dnt0 --limit 200 --output text
```

Important lesson: do not assume Render has deployed the latest `origin/main`.
Always check the live deploy commit before debugging production behavior. We saw
production serving an older commit even after `main` had newer fixes.

## Production Testing

Streamlit renders client-side, so `Invoke-WebRequest` only proves the service is
up. Use Playwright for real UI checks.

Cached Playwright Node package path used successfully:

```powershell
$env:NODE_PATH = "C:\Users\User\AppData\Local\npm-cache\_npx\420ff84f11983ee5\node_modules"
```

Production smoke-test flow that caught regressions:

1. Open `https://stockpredictors.onrender.com`.
2. Type `MSFT` into `Search for a stock ticker:`.
3. Blur/apply the input.
4. Click `Predict`.
5. Confirm rendered text contains `Predictions ready for MSFT`.

## Streamlit Widget Lessons

Text inputs:

- `del st.session_state[widget_key]` does not reliably clear a text input.
- `st.session_state[widget_key] = ""` also does not reliably clear it.
- Browser widget state can win on rerun.
- The reliable reset pattern is a counter in the widget key:

```python
search_key = f"new_ticker_input_{st.session_state.search_input_counter}"
new_ticker = st.text_input("Search for a stock ticker:", value="", key=search_key)

# On successful processing:
st.session_state.search_input_counter += 1
st.rerun()
```

Multiselect:

- `st.multiselect` renders before `search_and_add_ticker` runs.
- Appending to `selected_tickers` mid-render will not update the current widget.
- After autoselect, call `st.rerun()` so the multiselect is rebuilt with the new
  default value.
- Streamlit can fire a stale `on_change` callback from the previous multiselect
  value after widget recreation. Guard that stale callback with
  `pending_autoselect_ticker`; do not clear the guard until the stale callback is
  absorbed.

Search helper return contract:

- `True`: ticker was added/selected; increment counter and rerun.
- `None`: guard fired/already processed; clear bar without forced rerun.
- `False`: invalid or empty; leave bar so the warning remains visible.

## Search / yfinance Lessons

Do not validate searched ticker selection with yfinance before selecting it.
Render/yfinance can return empty data for a valid ticker due to transient
network/cache/rate-limit behavior. Search should add/select the ticker
deterministically; the prediction pipeline is the source of truth for invalid or
insufficient-data tickers.

The production regression was fixed by making search yfinance-independent:

```text
ad54c28 fix: select searched ticker without yfinance validation
```

Use `progress=False` on yfinance downloads when possible to avoid noisy progress
output in Streamlit/logs.

## yfinance Cache

Render yfinance cache path observed in production:

```text
/opt/render/.cache/py-yfinance
```

Cache reset can be done without restarting the service by removing/recreating
the py-yfinance cache directory. The app includes cache debug/clear controls and
tracing around:

- `debug_yfinance_cache_location()`
- `clear_yfinance_cache()`
- `invalidate_yfinance_cache()`

The yfinance retry logic watches for empty downloads plus shared yfinance error
markers such as `Failed download`, `Try after a while`, and `YFR...`.

## Market Status

Use `pandas_market_calendars` / XNYS for trading-day checks. Do not infer market
status from clock time alone.

Current display model:

- User-facing title is simple: `Market is Open`, `Market is Closed`, or
  `Market is Closed (Weekend/Holiday)`.
- Existing explanatory copy belongs in the subtitle.
- Non-trading days are internally treated as `AFTER_MARKET_CLOSE`, because the
  last market event was a close.

Manual market-state override:

```powershell
$env:MARKET_NOW = "2026-06-17 10:00"   # open
$env:MARKET_NOW = "2026-06-14 18:00"   # Saturday closed
$env:MARKET_NOW = ""                   # clear
```

## Streamlit Layout Lessons

Do not use unbalanced string HTML wrappers around separate Streamlit calls.
Streamlit renders each `st.markdown`/component separately and can auto-close the
HTML, producing an empty visual bar instead of a real container.

Use a native keyed container plus scoped CSS:

- `render_section_container(key, theme_name, ...)`
- `section_container_css(key, theme_name, ...)`
- `_safe_key(key)` for ticker symbols like `BRK-B` and `^GSPC`

The chart iframe cannot be nested inside a string-built `<div>` across separate
Streamlit calls.

## Plans And Archives

`UI_UX_PLAN.md` is complete and is the current source of truth for the finished
UI/UX work.

`archive/plans/DARK_MODE_PLAN.md` is historical. Extract durable details from it
only when needed; do not treat its pending-status lines as current.

`HANDOFF.md` is for session continuity and may become stale. Move durable
lessons into this file.

## Git / GitHub Notes

Normal `git push` has hung before because of local credential-helper behavior.
If that happens:

1. Check for stuck `git push` / `git-remote-https` processes.
2. Stop the stale processes.
3. Retry normal push.
4. If needed, use `gh api` with the authenticated GitHub CLI token as a fallback.

Do not force-push `main`.
