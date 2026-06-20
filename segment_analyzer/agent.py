from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .analyzer import AnalyzerConfig, SegmentAnalyzer
from .data_fetcher import StockDataClient, compact_date
from .indicators import add_moving_averages, normalize_ohlc
from .models import AnalysisRun, jsonable


@dataclass
class BatchResult:
    trade_date: str
    total: int
    runs: list[AnalysisRun]
    errors: list[dict[str, str]]


class SegmentAnalysisAgent:
    """Thin orchestration layer for data fetching, analysis and explanation."""

    def __init__(
        self,
        data_client: StockDataClient | None = None,
        analyzer: SegmentAnalyzer | None = None,
        config: AnalyzerConfig | None = None,
    ):
        self.config = config or AnalyzerConfig()
        self.data_client = data_client or StockDataClient()
        self.analyzer = analyzer or SegmentAnalyzer(self.config)

    def analyze_stock(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        direction: str = "auto",
        adjust: str = "qfq",
        name: str | None = None,
        scan_mode: str = "lowest",
    ) -> AnalysisRun:
        df = self.data_client.fetch_daily(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        reports = self._analyze(df, direction=direction, scan_mode=scan_mode)
        return AnalysisRun(
            symbol=symbol,
            name=name,
            trade_date=None,
            data_start=df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
            data_end=df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
            segment_count=len(reports),
            reports=reports,
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        direction: str = "auto",
        symbol: str | None = None,
        name: str | None = None,
        scan_mode: str = "lowest",
    ) -> AnalysisRun:
        clean = normalize_ohlc(df)
        reports = self._analyze(clean, direction=direction, scan_mode=scan_mode)
        return AnalysisRun(
            symbol=symbol,
            name=name,
            trade_date=None,
            data_start=clean["date"].min().strftime("%Y-%m-%d") if not clean.empty else None,
            data_end=clean["date"].max().strftime("%Y-%m-%d") if not clean.empty else None,
            segment_count=len(reports),
            reports=reports,
        )

    def analyze_limit_up_pool(
        self,
        trade_date: str,
        lookback_days: int = 520,
        direction: str = "up",
        adjust: str = "qfq",
        limit: int | None = None,
        scan_mode: str = "lowest",
    ) -> BatchResult:
        trade_date = compact_date(trade_date)
        stocks = self.data_client.fetch_limit_up_pool(trade_date)
        if limit:
            stocks = stocks[:limit]
        end_dt = datetime.strptime(trade_date, "%Y%m%d")
        start_date = (end_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")

        runs: list[AnalysisRun] = []
        errors: list[dict[str, str]] = []
        for stock in stocks:
            try:
                run = self.analyze_stock(
                    stock.code,
                    start_date=start_date,
                    end_date=trade_date,
                    direction=direction,
                    adjust=adjust,
                    name=stock.name,
                    scan_mode=scan_mode,
                )
                run.trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
                runs.append(run)
            except Exception as exc:
                errors.append({"symbol": stock.code, "name": stock.name or "", "error": str(exc)})
                if len(errors) >= self.config.batch_error_limit:
                    break
        return BatchResult(trade_date=trade_date, total=len(stocks), runs=runs, errors=errors)

    def write_json(self, payload: Any, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def explain_run(self, run: AnalysisRun) -> str:
        lines = []
        title = run.symbol or "CSV"
        if run.name:
            title += f" {run.name}"
        lines.append(f"{title}: {run.data_start} 至 {run.data_end}，线段数 {run.segment_count}")
        for report in run.reports:
            source = segment_source_label(report)
            lines.append(
                f"- #{report.segment_no} {report.segment_direction} [{source}]: {report.decision_status}; "
                f"{report.reasoning_summary}"
            )
        if run.errors:
            lines.append("错误: " + "；".join(run.errors))
        return "\n".join(lines)

    def _analyze(self, df: pd.DataFrame, direction: str, scan_mode: str):
        if scan_mode == "lowest":
            return self.analyzer.analyze_from_lowest(df)
        if scan_mode == "history":
            return self.analyzer.analyze_all(df, initial_direction=direction)
        raise ValueError(f"未知扫描模式: {scan_mode}")

    def explain_batch(self, batch: BatchResult) -> str:
        confirmed = sum(
            1 for run in batch.runs for report in run.reports if report.decision_status == "confirmed"
        )
        lines = [
            f"涨停池 {batch.trade_date}: 股票 {batch.total} 只，完成 {len(batch.runs)} 只，确认线段 {confirmed} 条，错误 {len(batch.errors)} 个。"
        ]
        for run in batch.runs[:10]:
            first = run.reports[0] if run.reports else None
            status = first.decision_status if first else "no_report"
            lines.append(f"- {run.symbol} {run.name or ''}: {status}")
        if len(batch.runs) > 10:
            lines.append(f"- 其余 {len(batch.runs) - 10} 只见 JSON 输出。")
        return "\n".join(lines)


def dataframe_with_ma_preview(df: pd.DataFrame) -> pd.DataFrame:
    return add_moving_averages(df)[["date", "open", "high", "low", "close", "ma5", "ma34"]]


def segment_source_label(report) -> str:
    if report.diagnostics.get("anchored_boundary_closure") or report.refresh_status == "anchored_boundary_closure":
        return "极值锚点"
    if report.pivot_confirm_k is not None:
        return "中枢确认"
    return "未确认来源"
