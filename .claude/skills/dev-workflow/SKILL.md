---
name: dev-workflow
description: StockPredictors fix/ship cycle — branch+worktree per change, local-gate tests, PR→review→merge→clean, Render deploy + browser-verify prod. Use when fixing a bug, shipping a change, or running the autonomous prod-debug loop.
---

# dev-workflow

Isolated, rollback-friendly cycle for every change. One change = one branch = one worktree = one PR.

## 0. Baseline (session start)
```bash
git fetch origin --quiet
git switch -c chore/$(date +%F)-<topic> main   # branch off production-parity main
```

## 1. Branch + worktree per fix
```bash
git worktree add /mnt/c/Users/User/StockPredictors-<branch> -b <branch> main
```
Sibling dir naming `StockPredictors-<branch>` matches CLAUDE.md. Worktree = isolation; rollback = just delete it.

## 2. Local gate (MUST pass before PR)
```bash
cd /mnt/c/Users/User/StockPredictors-<branch> && \
  /mnt/c/Users/User/StockPredictors/venv/Scripts/python.exe -m pytest -q
```
Use the **Windows venv python** — system python3 lacks app deps (pandas_market_calendars, streamlit). Baseline = 116 passed. Bash cwd resets each call, so `cd` in the same command.

## 3. PR → review → merge → clean
```bash
gh pr create --base main --head <branch> --title "..." --body "<root cause + test evidence>"
gh pr view <n> --json mergeable,mergeStateStatus   # expect MERGEABLE / CLEAN
gh pr comment <n> --body "<self-review: evidence>"
gh pr merge <n> --merge --delete-branch            # --merge keeps commits for cherry-pick
```
Worktree blocks local branch delete → remove it first, then prune:
```bash
git worktree remove /mnt/c/Users/User/StockPredictors-<branch> --force
git branch -D <branch>; git branch -f main origin/main
```

## 4. Deploy (Render = MANUAL, prod lags main)
```bash
CLI=/mnt/c/Users/User/.local/bin/render-cli/cli_v2.20.0.exe   # runs from WSL, pre-authed
SRV=srv-csvanmqj1k6c73c3dnt0
$CLI deploys list $SRV --output json    # check live commit FIRST (top status=live)
$CLI deploys create $SRV --confirm --output json
```
Poll `deploys list` until top `status=live` (~3-5 min). Run the poll in **background** (no foreground sleep).

## 5. Verify on prod
Logs: `$CLI logs --resources $SRV --limit 200 --output text` → grep `pipeline.ticker_exception`.
Browser (claude-in-chrome): open https://stockpredictors.onrender.com → select ticker → Predict →
wait (model retrains, slow) → confirm cards render, no skipped tickers / KeyError.

Rate-limit markers (`Try after a while`, `YFR`, `Failed download`) = account/IP throttle, not code.
Clear yfinance cache (`/opt/render/.cache/py-yfinance`) and retest.

## Rules
- Never force-push `main`. Never auto-delete prod data.
- Don't check in historical stock data (`*_data.csv`, `models/`, `known_tickers.json` are gitignored).
- One logical change per PR; PR body states root cause + test evidence.
