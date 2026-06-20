from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from segment_analyzer import AnalyzerConfig, SegmentAnalysisAgent
from segment_analyzer.models import jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="日线线段顶底识别程序")
    add_runtime_args(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    stock = sub.add_parser("stock", help="抓取并分析一只股票")
    add_runtime_args(stock)
    stock.add_argument("--symbol", required=True, help="股票代码，如 000001")
    stock.add_argument("--name", help="股票名称，可选")
    stock.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
    stock.add_argument("--end", default=datetime.now().strftime("%Y%m%d"), help="结束日期 YYYYMMDD")
    stock.add_argument("--direction", default="auto", choices=["auto", "up", "down", "both"], help="初始候选线段方向")
    stock.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权方式")

    limit_up = sub.add_parser("limit-up", help="批量分析某日全部涨停股票")
    add_runtime_args(limit_up)
    limit_up.add_argument("--date", required=True, help="交易日期 YYYYMMDD")
    limit_up.add_argument("--lookback-days", type=int, default=520, help="向前抓取多少自然日的日线")
    limit_up.add_argument("--direction", default="up", choices=["auto", "up", "down", "both"], help="初始候选线段方向")
    limit_up.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="复权方式")
    limit_up.add_argument("--limit", type=int, help="只分析前 N 只，调试用")

    csv = sub.add_parser("csv", help="分析本地 CSV 日线文件")
    add_runtime_args(csv)
    csv.add_argument("--file", required=True, help="CSV 文件，需含 date/open/high/low/close 或 AkShare 中文列")
    csv.add_argument("--symbol", help="股票代码，可选")
    csv.add_argument("--name", help="股票名称，可选")
    csv.add_argument("--direction", default="auto", choices=["auto", "up", "down", "both"], help="初始候选线段方向")

    gui = sub.add_parser("gui", help="打开股票线段分析 GUI 窗口")
    add_runtime_args(gui)

    return parser


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", default="data/cache", help="行情缓存目录，默认 data/cache")
    parser.add_argument("--no-cache", action="store_true", help="禁用行情缓存")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    parser.add_argument("--max-segments", type=int, default=200, help="最多连续识别多少条线段")
    parser.add_argument(
        "--scan-mode",
        default="lowest",
        choices=["lowest", "history"],
        help="lowest=从历史最低点向前后双向扫描；history=从历史左侧滑动发现",
    )


def main() -> None:
    args = build_parser().parse_args()
    config = AnalyzerConfig(max_segments=args.max_segments)
    agent = SegmentAnalysisAgent(config=config)
    agent.data_client.cache_dir = Path(args.cache_dir) if args.cache_dir else None
    agent.data_client.use_cache = not args.no_cache
    if agent.data_client.cache_dir:
        agent.data_client.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.command == "gui":
            from segment_analyzer.gui import run_gui

            run_gui(
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
                max_segments=args.max_segments,
                initial_scan_mode=args.scan_mode,
            )
            return
        if args.command == "stock":
            result = agent.analyze_stock(
                symbol=args.symbol,
                name=args.name,
                start_date=args.start,
                end_date=args.end,
                direction=args.direction,
                adjust=args.adjust,
                scan_mode=args.scan_mode,
            )
            print(agent.explain_run(result))
        elif args.command == "limit-up":
            result = agent.analyze_limit_up_pool(
                trade_date=args.date,
                lookback_days=args.lookback_days,
                direction=args.direction,
                adjust=args.adjust,
                limit=args.limit,
                scan_mode=args.scan_mode,
            )
            print(agent.explain_batch(result))
        elif args.command == "csv":
            df = pd.read_csv(args.file)
            result = agent.analyze_dataframe(
                df,
                direction=args.direction,
                symbol=args.symbol,
                name=args.name,
                scan_mode=args.scan_mode,
            )
            print(agent.explain_run(result))
        else:
            raise SystemExit(f"未知命令: {args.command}")
    except Exception as exc:
        error_payload = {"command": args.command, "status": "error", "error": str(exc)}
        if args.output:
            out = agent.write_json(error_payload, args.output)
            print(f"分析失败，错误已输出: {out}")
        print(f"分析失败: {exc}")
        raise SystemExit(1)

    if args.output:
        out = agent.write_json(result, args.output)
        print(f"JSON 已输出: {out}")
    else:
        print(json.dumps(jsonable(result), ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
