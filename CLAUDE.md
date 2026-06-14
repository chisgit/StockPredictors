# StockPredictors — Project Notes

## Running the app

Streamlit venv lives at `venv/Scripts/streamlit.exe` in the project root (`C:/Users/User/StockPredictors/venv/Scripts/streamlit.exe`).

Worktrees do **not** have their own venv — always launch via the main project venv, pointed at the worktree working directory:

```powershell
Start-Process -FilePath "C:\Users\User\StockPredictors\venv\Scripts\streamlit.exe" `
  -ArgumentList "run","stock_predictors.py","--server.port","8502" `
  -WorkingDirectory "C:\Users\User\StockPredictors-<branch-name>" `
  -WindowStyle Normal
```

Default port: `8501` (main), `8502+` (worktrees).

## Removing a worktree

Streamlit holds the worktree directory open. `git worktree remove` will fail with `Permission denied` if Streamlit is still running against it. Kill the process first:

```powershell
# Kill by port (replace 8502 with the port used)
netstat -ano | findstr ":8502" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
```

Then remove the worktree and delete the branch:

```bash
git worktree remove "C:/Users/User/StockPredictors-<branch-name>" --force
git branch -d <branch-name>
```
