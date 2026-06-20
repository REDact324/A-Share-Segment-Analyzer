from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .indicators import normalize_ohlc


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_ZT_POOL_URL = "https://push2ex.eastmoney.com/getTopicZTPool"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
EASTMONEY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/ztb/detail",
    "Accept": "application/json,text/plain,*/*",
}


@dataclass
class StockIdentity:
    code: str
    name: str | None = None


class StockDataClient:
    """Fetch daily A-share data.

    AkShare is the preferred source because it already wraps Eastmoney's
    historical K-line and limit-up pool endpoints. Historical K-lines have
    a direct Eastmoney fallback so a single-stock run can still work when
    AkShare's symbol map is temporarily stale.
    """

    def __init__(self, cache_dir: str | Path | None = "data/cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_daily(
        self,
        symbol: str,
        start_date: str = "20000101",
        end_date: str = "20500101",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        symbol = normalize_symbol(symbol)
        cache_path = self._cache_path("hist", f"{symbol}_{start_date}_{end_date}_{adjust or 'raw'}.csv")
        if self.use_cache and cache_path and cache_path.exists():
            return normalize_ohlc(pd.read_csv(cache_path))

        errors: list[str] = []
        try:
            import akshare as ak

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            out = normalize_ohlc(df)
        except Exception as exc:
            errors.append(f"AkShare stock_zh_a_hist 失败: {exc}")
            try:
                out = self._fetch_daily_eastmoney(symbol, start_date, end_date, adjust)
            except Exception as fallback_exc:
                errors.append(f"东方财富日线兜底失败: {fallback_exc}")
                try:
                    out = self._fetch_daily_tencent(symbol, start_date, end_date, adjust)
                except Exception as tencent_exc:
                    errors.append(f"腾讯日线兜底失败: {tencent_exc}")
                    raise RuntimeError("; ".join(errors)) from tencent_exc

        if cache_path:
            out.to_csv(cache_path, index=False)
        return out

    def fetch_limit_up_pool(self, trade_date: str) -> list[StockIdentity]:
        trade_date = compact_date(trade_date)
        cache_path = self._cache_path("limit_up", f"{trade_date}.csv")
        if self.use_cache and cache_path and cache_path.exists():
            return limit_up_records(pd.read_csv(cache_path))

        try:
            import akshare as ak

            df = ak.stock_zt_pool_em(date=trade_date)
        except Exception as exc:
            try:
                df = self._fetch_limit_up_pool_eastmoney(trade_date)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"获取涨停池失败。AkShare: {exc}; 东方财富兜底: {fallback_exc}"
                ) from fallback_exc

        if cache_path:
            df.to_csv(cache_path, index=False)
        return limit_up_records(df)

    def _fetch_daily_eastmoney(
        self, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        adjust_map = {"": "0", "qfq": "1", "hfq": "2", None: "0"}
        params: dict[str, Any] = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": adjust_map.get(adjust, "1"),
            "secid": f"{eastmoney_market(symbol)}.{symbol}",
            "beg": compact_date(start_date),
            "end": compact_date(end_date),
        }
        resp = requests.get(EASTMONEY_KLINE_URL, params=params, headers=EASTMONEY_HEADERS, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        klines = (payload.get("data") or {}).get("klines") or []
        if not klines:
            raise ValueError(f"东方财富没有返回 {symbol} 的日线数据")
        rows = []
        for line in klines:
            parts = line.split(",")
            rows.append(
                {
                    "date": parts[0],
                    "open": parts[1],
                    "close": parts[2],
                    "high": parts[3],
                    "low": parts[4],
                    "volume": parts[5],
                    "amount": parts[6],
                    "amplitude": parts[7],
                    "pct_change": parts[8],
                    "change": parts[9],
                    "turnover": parts[10] if len(parts) > 10 else None,
                }
            )
        return normalize_ohlc(pd.DataFrame(rows))

    def _fetch_daily_tencent(
        self, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        market_symbol = f"{tencent_market(symbol)}{symbol}"
        fq = {"qfq": "qfq", "hfq": "hfq", "": ""}.get(adjust, "qfq")
        rows = []
        target_start = pd.to_datetime(dashed_date(start_date))
        current_end = pd.to_datetime(dashed_date(end_date))

        for _ in range(80):
            if current_end < target_start:
                break
            param_parts = [
                market_symbol,
                "day",
                target_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
                "640",
            ]
            if fq:
                param_parts.append(fq)
            resp = requests.get(
                TENCENT_KLINE_URL,
                params={"param": ",".join(param_parts)},
                headers=EASTMONEY_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data")
            if not isinstance(data, dict):
                break
            stock_data = data.get(market_symbol) or {}
            klines = stock_data.get(f"{fq}day") or stock_data.get("day") or []
            if not klines:
                break
            for item in klines:
                if len(item) < 5:
                    continue
                rows.append(
                    {
                        "date": item[0],
                        "open": item[1],
                        "close": item[2],
                        "high": item[3],
                        "low": item[4],
                        "volume": item[5] if len(item) > 5 else None,
                        "amount": item[6] if len(item) > 6 else None,
                    }
                )
            earliest = pd.to_datetime(klines[0][0])
            if earliest <= target_start or len(klines) < 640:
                break
            current_end = earliest - pd.Timedelta(days=1)

        if not rows:
            raise ValueError(f"腾讯没有返回 {symbol} 的日线数据")
        return normalize_ohlc(pd.DataFrame(rows))

    def _fetch_limit_up_pool_eastmoney(self, trade_date: str) -> pd.DataFrame:
        params = {
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "dpt": "wz.ztzt",
            "Pageindex": "0",
            "pagesize": "10000",
            "sort": "fbt:asc",
            "date": trade_date,
        }
        resp = requests.get(EASTMONEY_ZT_POOL_URL, params=params, headers=EASTMONEY_HEADERS, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        pool = (payload.get("data") or {}).get("pool") or []
        rows = []
        for item in pool:
            rows.append(
                {
                    "代码": item.get("c"),
                    "名称": item.get("n"),
                    "最新价": item.get("p"),
                    "涨跌幅": item.get("zdp"),
                    "成交额": item.get("amount"),
                    "换手率": item.get("hs"),
                    "连板数": item.get("lbc"),
                    "首次封板时间": item.get("fbt"),
                    "最后封板时间": item.get("lbt"),
                    "封板资金": item.get("fund"),
                    "炸板次数": item.get("zbc"),
                    "所属行业": item.get("hybk"),
                }
            )
        return pd.DataFrame(rows)

    def _cache_path(self, category: str, filename: str) -> Path | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / category
        path.mkdir(parents=True, exist_ok=True)
        return path / filename


def limit_up_records(df: pd.DataFrame) -> list[StockIdentity]:
    mapping = {"代码": "code", "名称": "name", "股票代码": "code", "股票简称": "name"}
    out = df.rename(columns={c: mapping.get(c, c) for c in df.columns}).copy()
    if "code" not in out.columns:
        raise ValueError(f"涨停池数据缺少代码字段，实际字段: {list(df.columns)}")
    if "name" not in out.columns:
        out["name"] = None
    records: list[StockIdentity] = []
    for _, row in out.iterrows():
        records.append(StockIdentity(code=normalize_symbol(row["code"]), name=row.get("name")))
    return records


def normalize_symbol(symbol: object) -> str:
    digits = "".join(ch for ch in str(symbol).strip() if ch.isdigit())
    if not digits:
        raise ValueError(f"非法股票代码: {symbol!r}")
    return digits.zfill(6)[-6:]


def compact_date(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())[:8]


def eastmoney_market(symbol: str) -> str:
    # 1 = Shanghai, 0 = Shenzhen/Beijing in Eastmoney's public K-line API.
    return "1" if symbol.startswith(("5", "6", "9")) else "0"


def tencent_market(symbol: str) -> str:
    return "sh" if symbol.startswith(("5", "6", "9")) else "sz"


def dashed_date(value: str) -> str:
    compact = compact_date(value)
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
