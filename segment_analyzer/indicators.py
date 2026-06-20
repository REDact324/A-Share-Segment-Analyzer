from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close"}


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common AkShare/Eastmoney OHLC column names."""
    mapping = {
        "日期": "date",
        "时间": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    out = df.rename(columns={c: mapping.get(c, c) for c in df.columns}).copy()
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(f"OHLC 数据缺少字段: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out = out.reset_index(drop=True)
    return out


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_ohlc(df)
    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma34"] = out["close"].rolling(34, min_periods=34).mean()
    return out


def format_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")
