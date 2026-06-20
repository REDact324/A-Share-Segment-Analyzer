# Daily Segment Analyzer

Language: [中文](README.zh-CN.md) | English

This project is a Python tool for identifying daily stock price segments, confirmed tops/bottoms, pivots, turn candles, and unconfirmed candidates. It can run from the command line or through a Tkinter GUI.

For the full Chinese documentation and detailed rule definitions, see [README.zh-CN.md](README.zh-CN.md).

## Setup

```bash
pip install -r requirements.txt
```

Dependencies:

- `pandas`
- `requests`
- `akshare`

Daily market data is cached under `data/cache` by default. If a network endpoint is temporarily unavailable, the program can reuse existing cached data.

## Usage

Open the GUI:

```bash
python3 main.py gui
```

The GUI lets users enter a stock symbol, analyze all detected segments, see whether each segment comes from pivot confirmation or an extreme anchor, and export either a TXT summary or detailed JSON.

Analyze one stock:

```bash
python3 main.py stock --symbol 001259 --start 19900101 --end 20260529 --output out/001259.json
```

Analyze all limit-up stocks on a given day:

```bash
python3 main.py limit-up --date 20260515 --lookback-days 520 --direction up --output out/zt_20260515.json
```

Analyze a local CSV file:

```bash
python3 main.py csv --file data/sample.csv --direction both --output out/sample.json
```

CSV files may use `date/open/high/low/close`, or common AkShare Chinese column names such as `日期/开盘/最高/最低/收盘`.

Common options:

- `--scan-mode lowest`: start from the historical lowest low and complete segments forward/backward.
- `--scan-mode history`: scan from the left side of history in chronological order.
- `--direction auto|up|down|both`: initial direction for `history` mode.
- `--adjust ""|qfq|hfq`: adjustment mode; default is `qfq`.
- `--max-segments`: maximum number of consecutive segments.
- `--cache-dir` / `--no-cache`: control market data cache behavior.

## Project Structure

- `main.py`: command-line entry point.
- `segment_analyzer/data_fetcher.py`: daily market data and limit-up pool fetching.
- `segment_analyzer/indicators.py`: OHLC normalization and MA5/MA34 calculation.
- `segment_analyzer/analyzer.py`: core segment, pivot, turn candle, endpoint extension, and post-processing rules.
- `segment_analyzer/agent.py`: orchestration, batch tasks, JSON output, and summaries.
- `segment_analyzer/gui.py`: Tkinter GUI.
- `segment_analyzer/models.py`: output data structures.

## Output

Each JSON segment report includes fields such as:

- `segment_no`, `segment_direction`
- `confirmed_top`, `confirmed_bottom`
- `candidate_top`, `candidate_bottom`
- `unconfirmed_candidates`
- `pivot_confirm_k`, `pivot_line`, `pivot_lower`, `pivot_upper`, `pivot_type`
- `turn_k`, `initial_cross`, `refresh_status`
- `decision_status`

For complete field explanations and rule details, use the Chinese README: [README.zh-CN.md](README.zh-CN.md).
