# Caching Plan

## Status
| Item | Status |
|------|--------|
| CSV data cache (`fetch_data`) | ✅ done — ba2ec42 |
| Model cache (`train_model`) | ✅ done — ba2ec42 |
| `known_tickers.json` skip list | ✅ done — ba2ec42 |
| `models/` dir + gitignore | ✅ done — ba2ec42 |

---

## Design decisions (locked 2026-06-17)

### CSV data cache — `fetch_data()` at [data_handler.py:217](data_handler.py#L217)
- First fetch: pull full history 2008→now, save `{ticker}_data.csv`
- On Predict click: load CSV, check last row timestamp
  - Last row age < 30 s → use CSV as-is
  - Last row age ≥ 30 s → fetch delta since last row date, append to CSV
- Historical rows are **immutable** — never re-fetch past data
- 30 s threshold applies on Predict click only, not background polling

### Model cache — `train_model()` at [model.py:10](model.py#L10)
- After train: save model + scalers to `models/{ticker}_{date}_{model_type}.pkl` via joblib
- On Predict: if same-calendar-day file exists → load, skip retrain
- One retrain per ticker per calendar day max

### Ticker known-list — `known_tickers.json`
- After successful data fetch: record ticker in `known_tickers.json`
- Skip yfinance validation calls for tickers already in the list

---

## Scope

| File | Change |
|------|--------|
| [data_handler.py](data_handler.py) | `fetch_data()`: CSV save/load, delta fetch, staleness check |
| [model.py](model.py) | `train_model()`: joblib save/load, same-day TTL check |
| `models/` (new dir) | gitignore — not committed |
| `known_tickers.json` (new) | gitignore or commit empty stub `{}` |

---

## Acceptance criteria
- [ ] `AAPL` first Predict: fetches full history, writes CSV, trains model, writes pkl
- [ ] `AAPL` second Predict (< 30 s): loads CSV as-is, loads pkl, no retrain
- [ ] `AAPL` Predict after > 30 s: fetches delta only (not full history), appends CSV, loads pkl
- [ ] New calendar day: same CSV used, model retrained, new pkl written
- [ ] `known_tickers.json` written after first successful fetch
- [ ] pytest -q green

---

## Follow-ups (post-implementation)
- Cache eviction policy for old CSV / pkl files (none locked yet)
- Cache location override via env var (nice-to-have)
